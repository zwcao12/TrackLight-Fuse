import time
import torch
import torch.nn as nn
from tqdm import tqdm
from models.exact_features import MultimodalFeatures, ReconstrFeatures
from models.backbone_models import FeatureProjectionMLP, FeatureProjectionConv

if __name__ == "__main__":
    x = torch.randn(1,3,224,224).cuda()
    feature_extractor = MultimodalFeatures().eval().cuda()
    CFM_2Dto3D = FeatureProjectionConv(in_channels=256, out_channels=128).eval().cuda()
    CFM_3Dto2D = FeatureProjectionConv(in_channels=128, out_channels=256).eval().cuda()
    feature_reconstr = ReconstrFeatures().eval().cuda()
    time_list = []
    num = 500
    exclude_first = 10
    for i in tqdm(range(num)):
        torch.cuda.synchronize()
        tic = time.time()
        MultimodalFeatures(x)
        FeatureProjectionConv(in_channels=256, out_channels=128)
        FeatureProjectionConv(in_channels=128, out_channels=256)
        ReconstrFeatures()
        torch.cuda.synchronize()
        time_list.append(time.time()-tic)
    time_list = time_list[exclude_first:]
    print("FPS:{:.2f}".format(1/(sum(time_list)/(num-exclude_first))))
