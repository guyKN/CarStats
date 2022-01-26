from typing import List, Union

import torch

class ImprovedModel:
    """
    It seems that
    """
    def __init__(self, confidence_threshold = None, iou_threshold = None, target_names=None, class_agnostic: Union[bool, None] = True):
        if target_names is None:
            target_names = ["car", "bus", "truck"]
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5x").cuda()
        if class_agnostic is not None:
            # class_agnostic makes the model detect objects without looking at ther class
            self.model.agnostic = class_agnostic
        if confidence_threshold is not None:
            self.model.conf = confidence_threshold
        if iou_threshold is not None:
            self.model.iou = iou_threshold

        # Only look for classes whose name is in target_names
        self.model.classes = _find_indexes(self.model.names, target_names)

    def __call__(self, image):
        return self.model(image)



def load_model(confidence_threshold = None, iou_threshold = None, target_names=None, class_agnostic = None):
    return ImprovedModel(confidence_threshold, iou_threshold, target_names, class_agnostic)



"""
Returns the a list of all indexes of the list l that contains an element of values. 
"""
def _find_indexes(l:List, values:List)->List[int]:
    out = []
    for i, num in enumerate(l):
        if num in values:
            out.append(i)
    return out