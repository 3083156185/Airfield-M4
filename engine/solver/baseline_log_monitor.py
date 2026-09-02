import json
import math
import re   
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
   

_METRIC_PATTERN = re.compile(r"^([^\[\]]+)(?:\[(\d+)\])?$")
    

@dataclass 
class BaselineLogDecision:     
    active: bool
    should_stop: bool
    epoch: int
    metric: str 
    current: Optional[float] = None
    baseline: Optional[float] = None
    diff: Optional[float] = None
    bad_epochs: int = 0  
    patience: int = 1
    reason: str = ""


class BaselineLogMonitor: 
    def __init__(     
        self,
        baseline_by_epoch: Dict[int, Dict[str, Any]],
        metric: str = "test_coco_eval_bbox[0]",
        start_epoch: int = 0,
        min_ap_gap: float = 0.03,
        patience: int = 1,  
    ):    
        self.baseline_by_epoch = baseline_by_epoch  
        self.metric = metric
        self.start_epoch = max(0, int(start_epoch))
        self.min_ap_gap = float(min_ap_gap)
        self.patience = max(1, int(patience))
        self.bad_epochs = 0

    @classmethod  
    def from_config(cls, cfg: Dict[str, Any], total_epochs: int):    
        if not cfg or not cfg.get("enabled", False):     
            return None
   
        path = cfg.get("path")  
        if not path:    
            raise ValueError("baseline_log_monitor.path is required when enabled")

        start_ratio = float(cfg.get("start_ratio", 0.5)) 
        start_epoch = int(math.floor(int(total_epochs) * start_ratio)) 
        return cls(
            baseline_by_epoch=cls._load_baseline_log(path),   
            metric=cfg.get("metric", "test_coco_eval_bbox[0]"),  
            start_epoch=start_epoch,     
            min_ap_gap=cfg.get("min_ap_gap", 0.03), 
            patience=cfg.get("patience", 1), 
        )    

    @staticmethod
    def _load_baseline_log(path) -> Dict[int, Dict[str, Any]]:
        baseline_by_epoch = {} 
        log_path = Path(path)   
        if not log_path.exists(): 
            raise FileNotFoundError(f"Baseline log file does not exist: {log_path}")
   
        with log_path.open("r") as f:
            for line_no, line in enumerate(f, start=1):    
                line = line.strip()  
                if not line:
                    continue
                try:
                    record = json.loads(line)  
                except json.JSONDecodeError as exc:    
                    raise ValueError(f"Invalid JSON in baseline log {log_path}:{line_no}") from exc
  
                if "epoch" not in record: 
                    continue
                baseline_by_epoch[int(record["epoch"])] = record    

        return baseline_by_epoch
     
    def check(self, log_stats: Dict[str, Any], epoch: int) -> BaselineLogDecision: 
        epoch = int(epoch)  
        baseline_record = self.baseline_by_epoch.get(epoch) 
        current = self._extract_metric(log_stats, self.metric)
        baseline = None     
        diff = None  
        if baseline_record is not None:
            baseline = self._extract_metric(baseline_record, self.metric)
        if current is not None and baseline is not None: 
            diff = current - baseline    

        if epoch < self.start_epoch:
            return BaselineLogDecision(  
                active=False,
                should_stop=False,
                epoch=epoch,   
                metric=self.metric,     
                current=current,
                baseline=baseline,
                diff=diff,     
                bad_epochs=self.bad_epochs,  
                patience=self.patience,
                reason=f"waiting for start_epoch={self.start_epoch}",     
            )   
 
        if baseline_record is None:  
            return BaselineLogDecision(  
                active=True,
                should_stop=False,
                epoch=epoch, 
                metric=self.metric,
                bad_epochs=self.bad_epochs,     
                patience=self.patience,  
                reason="baseline epoch is missing",
            )
     
        if current is None or baseline is None:     
            return BaselineLogDecision(   
                active=True, 
                should_stop=False,   
                epoch=epoch,
                metric=self.metric,  
                current=current,
                baseline=baseline,
                bad_epochs=self.bad_epochs,
                patience=self.patience,  
                reason="metric is missing",   
            )

        if diff < -self.min_ap_gap:     
            self.bad_epochs += 1 
        else:
            self.bad_epochs = 0 
  
        should_stop = self.bad_epochs >= self.patience
        reason = ""     
        if should_stop: 
            reason = (  
                f"current AP is lower than baseline AP by more than "
                f"{self.min_ap_gap:.4f} for {self.bad_epochs} consecutive epochs"
            )  

        return BaselineLogDecision(
            active=True,
            should_stop=should_stop,
            epoch=epoch, 
            metric=self.metric,    
            current=current,
            baseline=baseline,  
            diff=diff,
            bad_epochs=self.bad_epochs,
            patience=self.patience, 
            reason=reason,
        )

    @staticmethod
    def _extract_metric(record: Dict[str, Any], metric: str) -> Optional[float]:    
        match = _METRIC_PATTERN.match(metric)     
        if match is None:
            raise ValueError(f"Unsupported metric selector: {metric}")     
     
        key, index = match.groups()
        value = record.get(key)
        if value is None:     
            return None
        if index is not None:
            try:  
                value = value[int(index)]     
            except (IndexError, TypeError):    
                return None
    
        try:     
            return float(value)
        except (TypeError, ValueError):
            return None
