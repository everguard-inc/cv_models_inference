from collections import OrderedDict
import time
import albumentations as A
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from typing import Any, Dict, List, NoReturn, Optional

from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2


class CombineImageClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        if 'resnet34' in config['model_name']:
            self.features = torchvision.models.resnet34(pretrained=False)
            self.in_features = self.features.fc.in_features
            self.fc = nn.Linear(self.in_features, config['num_classes'])
            self.features = torch.nn.Sequential(*(list(self.features.children())[:-2]))
            self.features.add_module(module = nn.AdaptiveAvgPool2d(output_size=(1, 1)), name = 'avgpool')
        else:
            raise Exception("expected resnet34 in config")

    def forward(self, x):
        features = self.features(x)
        features = features.view(x.shape[0],self.in_features)
        x = self.fc(features)
        return x, features
    
    
class RemoteHoldingClassifier():
    def __init__(self, config):
        self._load_cfg(config)
        self.__init_model__()
        
    def _load_cfg(self, config: Dict[str, Any]) -> NoReturn:
        self.config = config
        self.weights_path = self.config['weights_path']
        self.device = self.config['device']
        self.img_size = self.config['img_size']
        self._channel_mean = 255 * torch.tensor(
            self.config['channel_mean'], dtype=torch.float16).view(1, 3, 1, 1).to(self.device)
        self._channel_std = 255 * torch.tensor(
            self.config['channel_std'], dtype=torch.float16).view(1, 3, 1, 1).to(self.device)

    def __init_model__(self):
        self.model = CombineImageClassification(self.config)
        state_dict = torch.load(self.weights_path)
        state_dict_keys = list(state_dict.keys())
        features_module_state_dict = OrderedDict()
        fc_module_state_dict = OrderedDict()
        for ind, fmk in enumerate(list(self.model.features.state_dict().keys())):
            features_module_state_dict.update({fmk:state_dict[state_dict_keys[ind]]})
        for ind, fmk in enumerate(list(self.model.fc.state_dict().keys())):
            ind+=len(list(self.model.features.state_dict().keys()))
            fc_module_state_dict.update({fmk:state_dict[state_dict_keys[ind]]})
        self.model.features.load_state_dict(features_module_state_dict)
        self.model.fc.load_state_dict(fc_module_state_dict)
        self.model.half()
        self.model = self.model.to(self.device)
        self.model.eval()

    def forward(self, images, masks, bboxes, num_bboxes):
        batch = self.create_combine_images(images, masks, bboxes, num_bboxes)
        outputs, features = self.model(batch)
        features = features.detach().cpu().numpy()
        _, labels = torch.max(outputs.data, 1)
        labels = labels.cpu().numpy()
        chunk_sizes = list(map(lambda lst: len(lst), bboxes))
        labels_grouped, features_grouped = [], []
        start = 0
        for end in chunk_sizes:
            labels_grouped.append(labels[start:start+end])
            features_grouped.append(features[start:start+end])
            start += end
        return bboxes, labels_grouped, features_grouped

    def create_combine_images(self, images: np.ndarray, masks: np.ndarray, bboxes: np.ndarray, num_bboxes: int):
        masks = self.prepare_mask(masks)
        combine_images = np.zeros(
            (num_bboxes, images[0].shape[0], images[0].shape[1], 3)
        )
        crop_index = 0
        for image_ind, image_bboxes in enumerate(bboxes):
            for box in image_bboxes:
                combine_images[crop_index,box[1]:box[3], box[0]:box[2]] = images[image_ind, box[1]:box[3], box[0]:box[2]]
                combine_images[crop_index] += images[image_ind] * masks[image_ind]
                crop_index+=1
        combine_images = combine_images.transpose((0, 3, 1, 2))
        combine_images = torch.from_numpy(combine_images).to(torch.float16).to(self.device)
        combine_images = torch.nn.functional.interpolate(
            combine_images, size=(self.img_size), mode="bilinear", align_corners = True
        )
        combine_images = (combine_images - self._channel_mean) / self._channel_std
        return combine_images

    def prepare_mask(self, masks):
        masks = np.sum(masks, axis = 1).squeeze()
        masks[masks > 0] = 1
        masks_rgb = np.zeros((*masks.shape,3))
        for i in range(3):
            masks_rgb[:,:,:,i] = masks
        return masks_rgb