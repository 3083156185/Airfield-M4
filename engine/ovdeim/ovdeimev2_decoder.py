from typing import List
   
import torch
import torch.nn as nn
    
from ..core import register
from ..deim.deimv2_decoder import DEIMV2Transformer    
from ..deim.dfine_utils import distance2bbox, weighting_function   
from ..deim.denoising import get_contrastive_denoising_training_group    
from ._shared import inverse_sigmoid  
from .cls_embed import ClassEmbed

__all__ = ["OVDEIMV2Transformer"]    

class OVDEIMV2TransformerDecoder(nn.Module):
    def __init__(self, base_decoder: nn.Module) -> None:  
        super().__init__()
        self.hidden_dim = base_decoder.hidden_dim
        self.num_layers = base_decoder.num_layers  
        self.layer_scale = base_decoder.layer_scale
        self.num_head = base_decoder.num_head
        self.eval_idx = base_decoder.eval_idx
        self.up = base_decoder.up
        self.reg_scale = base_decoder.reg_scale  
        self.reg_max = base_decoder.reg_max
        self.layers = base_decoder.layers     
        self.lqe_layers = base_decoder.lqe_layers     
 
    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        value = value_proj(memory) if value_proj is not None else memory
        value = torch.nn.functional.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)     
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)     
        split_shape = [h * w for h, w in memory_spatial_shapes]   
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)  

    def convert_to_deploy(self):   
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)  
        self.layers = self.layers[: self.eval_idx + 1] 
        self.lqe_layers = nn.ModuleList([nn.Identity()] * self.eval_idx + [self.lqe_layers[self.eval_idx]])

    @staticmethod    
    def _run_score_head(head: nn.Module, output: torch.Tensor, text_feats: torch.Tensor):
        if isinstance(head, ClassEmbed):
            return head(output, text_feats)  
        return head(output)    

    def forward(
        self,  
        target,
        ref_points_unact,     
        memory,     
        spatial_shapes,
        text_feats,
        bbox_head,
        score_head,  
        query_pos_head,
        pre_bbox_head,  
        integral,     
        up,  
        reg_scale, 
        attn_mask=None,  
        memory_mask=None,
        dn_meta=None,
    ):
        output = target
        output_detach = pred_corners_undetach = 0     
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []  
        dec_out_logits = []
        dec_out_pred_corners = []   
        dec_out_refs = []
        project = weighting_function(self.reg_max, up, reg_scale) if not hasattr(self, "project") else self.project
   
        ref_points_detach = torch.sigmoid(ref_points_unact) 
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)   
 
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
   
            if i >= self.eval_idx + 1 and self.layer_scale > 1:   
                query_pos_embed = torch.nn.functional.interpolate(query_pos_embed, scale_factor=self.layer_scale)   
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = torch.nn.functional.interpolate(output, size=query_pos_embed.shape[-1])   
                output_detach = output.detach()
  
            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)     
 
            if i == 0:
                pre_bboxes = torch.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = None if not self.training and isinstance(score_head[0], nn.Identity) else \
                    self._run_score_head(score_head[0], output, text_feats)
                ref_points_initial = pre_bboxes.detach()
  
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach    
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)    

            if self.training or i == self.eval_idx:     
                scores = self._run_score_head(score_head[i], output, text_feats)
                scores = self.lqe_layers[i](scores, pred_corners)    
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox) 
                dec_out_pred_corners.append(pred_corners)  
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break
     
            pred_corners_undetach = pred_corners   
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()
    
        return (   
            torch.stack(dec_out_bboxes),     
            torch.stack(dec_out_logits),     
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),  
            pre_bboxes,
            pre_scores, 
        )  
    
    
@register() 
class OVDEIMV2Transformer(DEIMV2Transformer):
    __share__ = ["num_classes", "eval_spatial_size"]  

    def __init__(
        self,
        num_classes=80,    
        hidden_dim=256,    
        num_queries=300, 
        feat_channels=[512, 1024, 2048],     
        feat_strides=[8, 16, 32],   
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu", 
        num_denoising=100,     
        label_noise_ratio=0.5,
        box_noise_scale=1.0,     
        learn_query_content=False,
        eval_spatial_size=None,  
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True, 
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,  
        reg_scale=4.0,  
        layer_scale=1,     
        mlp_act="relu",
        use_gateway=True,   
        share_bbox_head=False,
        share_score_head=False,
    ):  
        assert query_select_method in ("default", "one2many"), "agnostic query selection is not supported in OVDEIMV2"
        super().__init__(     
            num_classes=num_classes,     
            hidden_dim=hidden_dim,  
            num_queries=num_queries,  
            feat_channels=feat_channels,
            feat_strides=feat_strides,  
            num_levels=num_levels,  
            num_points=num_points,
            nhead=nhead,    
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,  
            activation=activation,    
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learn_query_content=learn_query_content,     
            eval_spatial_size=eval_spatial_size,    
            eval_idx=eval_idx,
            eps=eps,  
            aux_loss=aux_loss,
            cross_attn_method=cross_attn_method, 
            query_select_method=query_select_method,    
            reg_max=reg_max, 
            reg_scale=reg_scale,     
            layer_scale=layer_scale,
            mlp_act=mlp_act,    
            use_gateway=use_gateway,  
            share_bbox_head=share_bbox_head,
            share_score_head=share_score_head,
        )

        self.enc_score_head = ClassEmbed()
        self.dec_score_head = nn.ModuleList([ClassEmbed() for _ in range(self.num_layers)])  
        self.decoder = OVDEIMV2TransformerDecoder(self.decoder)     

    @staticmethod     
    def _flatten_targets(targets: List[dict]) -> dict:
        labels, boxes, idx, num_boxes = [], [], [], []
        for batch_idx, target in enumerate(targets):    
            labels.append(target["labels"])
            boxes.append(target["boxes"])   
            idx.append(     
                torch.full((target["labels"].shape[0],), batch_idx, dtype=torch.int64, device=target["labels"].device)
            )   
            num_boxes.append(int(target["labels"].shape[0]))
   
        device = targets[0]["labels"].device
        return {
            "idx": torch.cat(idx, dim=0) if idx else torch.empty((0,), dtype=torch.int64, device=device),   
            "labels": torch.cat(labels, dim=0) if labels else torch.empty((0,), dtype=torch.int64, device=device),
            "boxes": torch.cat(boxes, dim=0) if boxes else torch.empty((0, 4), dtype=torch.float32, device=device),
            "num_boxes": num_boxes, 
        }
  
    @staticmethod
    def _prepare_text_feats(text_feats: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        if text_feats.dim() == 2:
            text_feats = text_feats.unsqueeze(0)     
        text_feats = text_feats.to(device)
        if text_feats.shape[0] == 1 and batch_size > 1:     
            text_feats = text_feats.repeat(batch_size, 1, 1)
        if text_feats.shape[0] != batch_size:
            raise ValueError(f"text_feats batch mismatch: got {text_feats.shape[0]}, expected {batch_size}")  
        return text_feats    

    def _get_decoder_input(     
        self, 
        memory: torch.Tensor,
        spatial_shapes,
        text_feats: torch.Tensor,
        denoising_logits=None,
        denoising_bbox_unact=None,
    ):    
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device) 
        else:   
            anchors = self.anchors     
            valid_mask = self.valid_mask
        if memory.shape[0] > 1 and anchors.shape[0] == 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
     
        memory = valid_mask.to(memory.dtype) * memory
        enc_outputs_logits = self.enc_score_head(memory, text_feats)     
  
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(   
            memory, enc_outputs_logits, anchors, self.num_queries
        )
        enc_topk_bbox_unact = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors     
     
        enc_topk_bboxes_list, enc_topk_logits_list = [], [] 
        if self.training:   
            enc_topk_bboxes_list.append(torch.sigmoid(enc_topk_bbox_unact))   
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])    
        else:   
            content = enc_topk_memory.detach()
   
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)    

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list
 
    def forward(self, feats, targets=None, text_feats=None):
        if text_feats is None:
            raise RuntimeError("OVDEIMV2Transformer requires text_feats.")

        memory, spatial_shapes = self._get_encoder_input(feats)     
        text_feats = self._prepare_text_feats(text_feats, memory.shape[0], memory.device)

        if self.training and self.num_denoising > 0:   
            if targets is None:
                raise RuntimeError("Training with denoising requires targets.")   
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,   
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,     
                box_noise_scale=self.box_noise_scale,
            )
        else:    
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
   
        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = self._get_decoder_input(  
            memory,
            spatial_shapes,
            text_feats,   
            denoising_logits,   
            denoising_bbox_unact,    
        )   

        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            init_ref_contents,   
            init_ref_points_unact, 
            memory,    
            spatial_shapes,
            text_feats,  
            self.dec_bbox_head,
            self.dec_score_head,   
            self.query_pos_head,    
            self.pre_bbox_head,
            self.integral,  
            self.up,  
            self.reg_scale,    
            attn_mask=attn_mask,    
            dn_meta=dn_meta,
        )
  
        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1) 
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)     
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)
 
        if self.training: 
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],   
                "up": self.up, 
                "reg_scale": self.reg_scale,
            }  
        else:
            out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}   
   
        if self.training and self.aux_loss:    
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],    
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],    
                out_logits[-1],  
            )   
            out["enc_aux_outputs"] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": False}
 
            if dn_meta is not None:    
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,  
                    dn_out_refs,   
                    dn_out_corners[-1],
                    dn_out_logits[-1],   
                )     
                out["dn_pre_outputs"] = {"pred_logits": dn_pre_logits, "pred_boxes": dn_pre_bboxes}
                out["dn_meta"] = dn_meta 

        return out     
