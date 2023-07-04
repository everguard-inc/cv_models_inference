import os
from typing import Any, List, NoReturn, Sequence, Dict

import cv2
import numpy as np
import torch
from easydict import EasyDict
from deep_models.deep_detector import DeepSegmentor

class MultilabelSegmentator(DeepSegmentor):
    def _load_once_cfg(self, global_config: EasyDict, local_config: EasyDict, **kwargs: Any) -> NoReturn:
        super()._load_once_cfg(global_config, local_config, **kwargs)
        self._threshold = local_config.threshold
        self._output_size = local_config.get("output_size", self._input_size)
        self._class_names = local_config.get("class_names")
        self._num_classes = len(self._class_names)
        self._class_colors = local_config.get("class_colors")
        self._device = local_config.get("device", "cuda:0")
        self._gpu_preprocess = local_config.get("preprocess_with_gpu", True)
        self._torch_precision = self._string_to_torch_dtype(self._precision)
        self._channel_mean = 255 * torch.tensor(local_config.channel_mean, dtype=self._torch_precision).view(
            1, 3, 1, 1
        )
        self._channel_std = 255 * torch.tensor(local_config.channel_std, dtype=self._torch_precision).view(1, 3, 1, 1)
        self._cpu_additional = False
        self._batch_size = local_config["batch_size"]
        self._tensor_buffer = torch.empty(
            size = (self._batch_size, 3, self._input_size[0], self._input_size[1]), 
            dtype = self._torch_precision, device = self._device
        )

    def load_cfg(self, config: EasyDict) -> EasyDict:
        cfg = super().load_cfg(config)
        self._batch_size = cfg['batch_size']
        self.set_preprocess_key(resize_mode="cv", output_dtype=None)
        return cfg

    def _string_to_torch_dtype(self, string: str) -> torch.dtype:
        string = string.lower()
        if string == "fp32":
            return torch.float32
        elif string == "fp16":
            return torch.float16
        elif string == "int8":
            return torch.int8
        else:
            raise ValueError(f"Unsupportable precision: {string}")

    def load_pb(self) -> NoReturn:
        print(f"Loading MultilabelSegmentatior model from {os.path.basename(self._pb_file)} to {self._device}")
        if os.path.splitext(self._pb_file)[1] not in [".engine", ".plan", ".trt"]:
            self._model = torch.jit.load(self._pb_file).to(self._device).to(self._torch_precision).eval()  # type: ignore
            self._model.eval()
        else:
            from cv_models_inference.src.eg_utils.eg_utils.convertation.trt_model import TRTModel  # type: ignore

            self._model = TRTModel(model_path=self._pb_file, device=str(self._device))

        if self._gpu_preprocess:
            self._channel_mean = self._channel_mean.to(self._device)
            self._channel_std = self._channel_std.to(self._device)

        self._warmup()

    def _warmup(self) -> NoReturn:
        print("Warm up...")
        check_array = np.random.random((1,3,*self._input_size[:2]))
        self._tensor_buffer = torch.from_numpy(check_array).to(self._device).to(self._torch_precision)
        self.predict_on_batch()
        print("Done")

    def gpu_batch_preprocess(self) -> torch.Tensor:
        self._tensor_buffer = torch.nn.functional.interpolate(
            self._tensor_buffer, size=(self._input_size[:2]), mode="bilinear", align_corners = True
        )
        self._tensor_buffer = (self._tensor_buffer - self._channel_mean) / self._channel_std

    def predict_on_batch(self) -> np.ndarray:
        output = self._model(self._tensor_buffer)
        masks = (output.sigmoid() > self._threshold).to(torch.uint8).cpu().numpy()
        return masks

    def _postprocess_image_masks(self, image_masks: np.ndarray) -> np.ndarray:
        if self._output_size[:2] != self._input_size[:2]:
            image_masks = torch.nn.functional.interpolate(
                image_masks, size=(self._output_size[:2]), mode="bilinear", align_corners=True
            )
        return image_masks

    # methods for testing
    def forward_batch(self, batch: np.ndarray) -> List[np.ndarray]:
        batch = batch.transpose((0, 3, 1, 2))
        self._tensor_buffer = torch.from_numpy(batch).to(self._device).to(self._torch_precision)
        self.gpu_batch_preprocess()
        masks = self.predict_on_batch()
        output = self._postprocess_image_masks(image_masks=masks)
        return output

    def forward_image(self, image: np.ndarray) -> np.ndarray:
        return self.forward_batch([image])[0]
