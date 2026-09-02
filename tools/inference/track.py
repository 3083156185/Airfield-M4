import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np

try:
    from boxmot import ByteTrack, StrongSort
except ImportError as e:
    pass

class Track:
    def __init__(self, det_thresh=0.3) -> None:
        self.track = None
        self.track = ByteTrack(det_thresh=det_thresh)
    
    def transform_mot(self, classes, boxes, scores):
        mot_result = []
        for cls, box, sco in zip(classes.cpu().tolist()[0], boxes.cpu().tolist()[0], scores.cpu().tolist()[0]):
            mot_result.append(box + [float(sco), int(cls)])
        return np.array(mot_result)

    def __call__(self, image, results):
        mot_results = self.transform_mot(*results)
        track_results = self.track.update(mot_results, image)
        return torch.from_numpy(track_results)