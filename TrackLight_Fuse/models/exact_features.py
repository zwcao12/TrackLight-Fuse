import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from utils.metrics_utils import calculate_au_pro
from models.backbone_models import EncoderRGB, EncoderD
from models.backbone_models import DecoderRGB, DecoderD


class MultimodalFeatures(torch.nn.Module):
    def __init__(self, image_size = 224):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.deep_feature_extractor_rgb = EncoderRGB()
        self.deep_feature_extractor_d = EncoderD()

        self.deep_feature_extractor_rgb.to(self.device)
        self.deep_feature_extractor_d.to(self.device)

        self.image_size = image_size

        # * Applies a 2D adaptive average pooling over an input signal composed of several input planes. 
        # * The output is of size H x W, for any input size. The number of output features is equal to the number of input planes.
        self.resize = torch.nn.AdaptiveAvgPool2d((224, 224))
        
        self.average = torch.nn.AvgPool2d(kernel_size = 3, stride = 1) 

    def forward(self, rgb, d):
        rgb = rgb.to(self.device)
        d = d.to(self.device)

        # with torch.no_grad():
        rgb_feature_maps = self.deep_feature_extractor_rgb(rgb)
        d_feature_maps = self.deep_feature_extractor_d(d)

        d_feature_maps = [fmap for fmap in [d_feature_maps]]
        rgb_feature_maps = [fmap for fmap in [rgb_feature_maps]]

        return rgb_feature_maps, d_feature_maps

    def calculate_metrics(self):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)

        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)

    def get_features_maps(self, rgb, d):
        rgb_feature_maps, d_feature_maps = self(rgb, d)
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        d_patch = torch.cat(d_feature_maps, 1)

        return rgb_patch, d_patch

class ReconstrFeatures(torch.nn.Module):
    def __init__(self, image_size = 224):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.deep_feature_reconstr_rgb = DecoderRGB(3, 0, 224)
        self.deep_feature_reconstr_d = DecoderD(3, 0, 224)

        self.deep_feature_reconstr_rgb.to(self.device)
        self.deep_feature_reconstr_d.to(self.device)

        self.image_size = image_size

    def forward(self, rgb_feature, d_feature):
        rgb_feature = rgb_feature.to(self.device)
        d_feature = d_feature.to(self.device)
        # with torch.no_grad():

        rgb_re = self.deep_feature_reconstr_rgb(rgb_feature)
        d_re = self.deep_feature_reconstr_d(d_feature)

        return rgb_re, d_re