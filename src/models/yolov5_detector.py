import os
import time
from collections import defaultdict
from typing import Any, Dict, List, NoReturn, Optional, Tuple

import numpy as np
import torch
import numpy as np
from torch import jit
from yolov5.utils.augmentations import letterbox
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import check_img_size, non_max_suppression
from eg_utils.helpers.bboxes import scale_coords_batch

class YoloV5Detector:
    def __init__(self, config):
        self._load_cfg(config)
        self._load_model()
        
    def _load_cfg(self, config: Dict[str, Any]) -> NoReturn:
        self._config = config
        self._model_path = config['model_path']
        self._imgsz = config["imgsz"]
        self._dnn = config["dnn"]
        self._half : bool = config["fp16"]
        self._coco_data = config["coco_data"]
        self._device = select_device(config["device"])
        self._conf_thres = config["conf_thres"]
        self._iou_thres = config["iou_thres"]
        self._agnostic_nms = config.get("agnostic_nms", True)
        self._max_det = config.get("max_det", 200)
        self._timers: Dict[str, List[float]] = defaultdict(list)

    def _load_model(self) -> NoReturn:
        self._model = DetectMultiBackend(
                            self._model_path, dnn=self._dnn,
                            device=self._device, data=self._coco_data
                            )
        pt = self._model.pt
        stride = self._model.stride
        jit = self._model.jit
        onnx = self._model.onnx
        engine = self._model.engine
        
        self._imgsz = check_img_size(self._imgsz, s=stride)
        self._half &= (pt or jit or onnx or engine) and self._device.type != 'cpu'
        if pt or jit:
            self._model.model.half() if self._half else self._model.model.float()
        self._warmup()

    def _warmup(self) -> NoReturn:
        self._model.warmup(imgsz=(1 if self._model.pt else bs, 3, *self._imgsz), half=self._half)  # warmup
            
    def _preprocess_batch(self, images: torch.Tensor) -> torch.Tensor:
        batch = []
        for image in images:
            image = letterbox(image, self._imgsz, stride=self._model.stride, auto=self._model.pt)[0]
            batch.append(image)
        batch = np.stack(batch)
        batch = batch.transpose((0, 3, 1, 2))
        batch = np.ascontiguousarray(batch)
        batch = torch.from_numpy(batch).to(self._device)
        batch = batch.half() if self._half else batch.float()  # uint8 to fp16/32
        batch /= 255  # 0 - 255 to 0.0 - 1.0
        return batch

    def forward_image(self, image: torch.Tensor) -> np.ndarray:
        output = self.forward_batch(image)[0]
        return output

    def forward_batch(self, images: torch.Tensor) -> List[np.ndarray]:
        batch = self._preprocess_batch(images)
        bboxes = self._model(batch, augment=False)        
        bboxes = torch.stack(non_max_suppression(
            bboxes, self._conf_thres, self._iou_thres, None, self._agnostic_nms, max_det=self._max_det
            ))
        if len(bboxes):
            bboxes = scale_coords_batch(batch.shape[2:], bboxes, images.shape[1:3]).round()
        bboxes = bboxes.cpu().detach().numpy()
        return bboxes