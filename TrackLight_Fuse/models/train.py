import argparse
import os
import torch
import torch.nn as nn
import wandb
import numpy as np
from itertools import chain
from tqdm import tqdm, trange
import torch.nn.functional as F
from models.exact_features import MultimodalFeatures, ReconstrFeatures
from models.dataset import get_data_loader
from models.backbone_models import FeatureProjectionConv

def set_seeds(sid=115):
    np.random.seed(sid)

    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)

def train_CFM(args):

    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f'{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'

    wandb.init(
        project = 'crossmodal-feature-mappings',
        name = model_name
    )

    '''Dataloader.'''
    train_loader = get_data_loader("train", class_name = args.class_name, img_size = 224, dataset_path = args.dataset_path, batch_size = args.batch_size, shuffle = True)

    '''Model instantiation.'''
    feature_extractor = MultimodalFeatures()

    CFM_2Dto3D = FeatureProjectionConv(in_channels=256, out_channels=128)
    CFM_3Dto2D = FeatureProjectionConv(in_channels=128, out_channels=256)

    feature_reconstr = ReconstrFeatures()

    optimizer = torch.optim.Adam(params = chain(CFM_2Dto3D.parameters(), CFM_3Dto2D.parameters(), feature_extractor.parameters(), feature_reconstr.parameters()))

    CFM_2Dto3D.to(device), CFM_3Dto2D.to(device)
    feature_extractor.to(device), feature_reconstr.to(device)
    metric = torch.nn.CosineSimilarity(dim = -1, eps = 1e-06)

    for epoch in trange(args.epochs_no, desc = f'Training Feature Transfer Net.'):

        epoch_cos_sim_3Dto2D, epoch_cos_sim_2Dto3D = [], []
        epoch_cos_sim_rgbtorgbre, epoch_cos_sim_dtodre = [], []

        '''------------ [Trainig Loop] ------------ '''

        '''* Return (rgb_img, d_img), globl_label'''
        for (rgb, d), _ in tqdm(train_loader, desc = f'Extracting feature from class: {args.class_name}.'):
            rgb, d = rgb.to(device), d.to(device)

            '''Make trainable.'''
            feature_extractor.train(), CFM_2Dto3D.train(), CFM_3Dto2D.train(), feature_reconstr.train()

            if args.batch_size == 1:
                rgb_patch, d_patch = feature_extractor.get_features_maps(rgb, d)
            else:
                rgb_patches = []
                d_patches = []
                rgb_patches = torch.tensor(rgb_patches).to(device)
                d_patches = torch.tensor(d_patches).to(device)
                rgb_patches_ori = []
                d_patches_ori = []
                rgb_patches_ori = torch.tensor(rgb_patches_ori).to(device)
                d_patches_ori = torch.tensor(d_patches_ori).to(device)

                for i in range(rgb.shape[0]):
                    rgb_patch, d_patch = feature_extractor.get_features_maps(rgb[i].unsqueeze(dim=0), d[i].unsqueeze(dim=0)) #[1,256,26,26] [1,128,26,26]
                    rgb_patch_re = F.interpolate(rgb_patch, size=(224, 224), mode="bilinear") #[1,256,224,224]
                    d_patch_re = F.interpolate(d_patch, size=(224, 224), mode="bilinear")  # [1,128,224,224]

                    rgb_patches = torch.cat((rgb_patches, rgb_patch_re), dim=0) #[batch,256,224,224]
                    d_patches = torch.cat((d_patches, d_patch_re), dim=0) #[batch,128,224,224]

                    rgb_patches_ori = torch.cat((rgb_patches_ori, rgb_patch), dim=0)  # [batch,256,26,26]
                    d_patches_ori = torch.cat((d_patches_ori, d_patch), dim=0)  # [batch,128,26,26]

            """ Predictions."""
            rgb_feat_pred = CFM_3Dto2D(d_patches) #mlp[224, 224, 512] conv [i+1,256,224,224]
            d_feat_pred = CFM_2Dto3D(rgb_patches) #mlp[224, 224, 256] conv [i+1,128,224,224]
            rgb_re, d_re = feature_reconstr(rgb_patches_ori, d_patches_ori)  # [2,3,26,26] [2,3,26,26] 2

            ''' Losses.'''
            d_mask = ((d_patches.sum(dim=(0, 1), keepdim=False)) == 0)  # [224,224]
            rgb_patches = rgb_patches.view(-1, 224, 224) #[512,224,224]
            d_patches = d_patches.view(-1, 224, 224) #[256,224,224]
            rgb_feat_pred = rgb_feat_pred.contiguous().view(-1, 224, 224) #[512,224,224]
            d_feat_pred = d_feat_pred.contiguous().view(-1, 224, 224) #[256,224,224]

            rgb_patches = rgb_patches.permute(1, 2, 0)  # [224,224,512]
            d_patches = d_patches.permute(1, 2, 0)  # [224,224,256]
            rgb_feat_pred = rgb_feat_pred.permute(1, 2, 0) # [224,224,512]
            d_feat_pred = d_feat_pred.permute(1, 2, 0) # [224,224,256]

            loss_3Dto2D = 1 - metric(d_feat_pred[~d_mask], d_patches[~d_mask]).mean()
            loss_2Dto3D = 1 - metric(rgb_feat_pred[~d_mask], rgb_patches[~d_mask]).mean()

            loss_rgbtorgbre = 1 - metric(rgb, rgb_re).mean()
            loss_dtodre = 1 - metric(d, d_re).mean()

            loss = loss_3Dto2D + loss_2Dto3D + loss_rgbtorgbre + loss_dtodre

            cos_sim_3Dto2D, cos_sim_2Dto3D = 1 - loss_3Dto2D.cpu(), 1 - loss_2Dto3D.cpu()
            cos_sim_rgbtorgbre, cos_sim_dtodre = 1 - loss_rgbtorgbre.cpu(), 1 - loss_dtodre.cpu()

            epoch_cos_sim_3Dto2D.append(cos_sim_3Dto2D), epoch_cos_sim_2Dto3D.append(cos_sim_2Dto3D)
            epoch_cos_sim_rgbtorgbre.append(cos_sim_rgbtorgbre), epoch_cos_sim_dtodre.append(cos_sim_dtodre)

            # Logging.
            wandb.log({
                "train/loss_3Dto2D" : loss_3Dto2D,
                "train/loss_2Dto3D" : loss_2Dto3D,
                "train/cosine_similarity_3Dto2D" : cos_sim_3Dto2D,
                "train/cosine_similarity_2Dto3D" : cos_sim_2Dto3D,
                "train/loss_rgbtorgbre": loss_rgbtorgbre,
                "train/loss_dtodre": loss_dtodre,
                "train/cosine_similarity_rgbtorgbre": cos_sim_rgbtorgbre,
                "train/cosine_similarity_dtodre": cos_sim_dtodre,
                })

            if (torch.isnan(loss_3Dto2D) or torch.isinf(loss_3Dto2D) or torch.isnan(loss_2Dto3D) or torch.isinf(loss_2Dto3D)
                    or torch.isnan(loss_rgbtorgbre) or torch.isinf(loss_rgbtorgbre) or torch.isnan(loss_dtodre) or torch.isinf(loss_dtodre)) :
                exit()

            # Optimization.
            if (not torch.isnan(loss_3Dto2D) and not torch.isinf(loss_3Dto2D) and not torch.isnan(loss_2Dto3D) and not torch.isinf(loss_2Dto3D) and
                    not torch.isnan(loss_rgbtorgbre) and not torch.isinf(loss_rgbtorgbre) and not torch.isnan(loss_dtodre) and not torch.isinf(loss_dtodre)):
                
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

        # Global logging.
        wandb.log({
            "global_train/cos_sim_3Dto2D" : torch.Tensor(epoch_cos_sim_3Dto2D, device = 'cpu').mean(),
            "global_train/cos_sim_2Dto3D" : torch.Tensor(epoch_cos_sim_2Dto3D, device = 'cpu').mean(),
            "global_train/cos_sim_rgbtorgbre": torch.Tensor(epoch_cos_sim_rgbtorgbre, device='cpu').mean(),
            "global_train/cos_sim_dtodre": torch.Tensor(epoch_cos_sim_dtodre, device='cpu').mean()
            })

    # Model saving.
    directory = f'{args.checkpoint_savepath}/{args.class_name}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(CFM_2Dto3D.state_dict(), os.path.join(directory, 'CFM_2Dto3D_' + model_name + '.pth'))
    torch.save(CFM_3Dto2D.state_dict(), os.path.join(directory, 'CFM_3Dto2D_' + model_name + '.pth'))
    torch.save(feature_extractor.state_dict(), os.path.join(directory, 'feature_extractor_' + model_name + '.pth'))
    torch.save(feature_reconstr.state_dict(), os.path.join(directory, 'feature_reconstr_' + model_name + '.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train Crossmodal Feature Networks (CFMs) on a dataset.')

    parser.add_argument('--dataset_path', default = './datasets/MVTec3D', type = str,
                        help = 'Dataset path.')

    parser.add_argument('--checkpoint_savepath', default = './checkpoints/mvtec', type = str,
                        help = 'Where to save the model checkpoints.')
    
    parser.add_argument('--class_name', default = 'peach', type = str, choices = ["fastener", "bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire",
                                                                               'CandyCane', 'ChocolateCookie', 'ChocolatePraline', 'Confetto', 'GummyBear', 'HazelnutTruffle', 'LicoriceSandwich', 'Lollipop', 'Marshmallow', 'PeppermintCandy'],
                        help = 'Category name.')
    
    parser.add_argument('--epochs_no', default = 50, type = int,
                        help = 'Number of epochs to train the CFMs.')

    parser.add_argument('--batch_size', default = 2, type = int,
                        help = 'Batch dimension. Usually 16 is around the max.')
    
    args = parser.parse_args()
    train_CFM(args)