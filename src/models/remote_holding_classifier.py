from collections import OrderedDict

import albumentations as A
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
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
        self.config = config
        self.__init_model__()

    def __init_model__(self):
        self.model = CombineImageClassification(self.config)
        state_dict = torch.load(self.config['weights_path'])
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
        self.model = self.model.to(self.config['device'])
        self.model.eval()

    def forward(self, images, masks, bboxes):
        bboxes = bboxes.tolist()
        batch = self.create_combine_images(self.config, image, masks, bboxes)
        outputs, features = self.model(batch)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().numpy()
        return bboxes, predicted, features.detach().cpu().numpy()

    def create_combine_images(self, config, images, masks, bboxes):
        masks = self.prepare_mask(masks)
        combine_images = torch.zeros(*images.shape)
        for i in range(len(combine_images)):
            for box in bboxes:
                combine_images[i, :, box[1]:box[3], box[0]:box[2]] = images[i, :, box[1]:box[3], box[0]:box[2]]
            combine_images[i] += images[i] * masks[i]
        combine_images = torch.nn.functional.interpolate(combine_images, size=(self._input_size[:2]), mode="bilinear")
        combine_images = (combine_images - self.config['channel_mean']) / self.config['channel_std']
        return combine_images

    def prepare_mask(self, masks):
        masks = torch.sum(masks, axis = 1)
        masks[masks > 0] = 1
        return masks