import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_utils import *
from torch.utils.data import DataLoader
import numpy as np
from utils.general_utils import SquarePad

def eyecandies_classes():
    return [
        'CandyCane',
        'ChocolateCookie',
        'ChocolatePraline',
        'Confetto',
        'GummyBear',
        'HazelnutTruffle',
        'LicoriceSandwich',
        'Lollipop',
        'Marshmallow',
        'PeppermintCandy',   
    ]

def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]
def fastener_classes():
    return [
        "fastener",
    ]

RGB_SIZE = 224

class BaseAnomalyDetectionDataset(Dataset):
    def __init__(self, split, class_name, img_size, dataset_path):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]

        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split)
        
        self.rgb_transform = transforms.Compose([
            # SquarePad(),
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation = transforms.InterpolationMode.BICUBIC), #resize
            transforms.ToTensor(), #transfer to tensor
            transforms.Normalize(mean = self.IMAGENET_MEAN, std = self.IMAGENET_STD) #guiyihua
            ])


class TrainValDataset(BaseAnomalyDetectionDataset):
    def __init__(self, split, class_name, img_size, dataset_path):
        super().__init__(split = split, class_name = class_name, img_size = img_size, dataset_path = dataset_path)

        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        rgb_paths.sort()
        tiff_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img = Image.open(rgb_path).convert('RGB')

        img = self.rgb_transform(img)
        img_d = read_tiff_organized_pc(tiff_path)
        img_d = self.rgb_transform(img_d)

        return (img, img_d), label

class TestDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(split = "test", class_name = class_name, img_size = img_size, dataset_path = dataset_path)

        self.gt_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                rgb_paths.sort()
                tiff_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))

                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)
        img_d = read_tiff_organized_pc(tiff_path)
        img_d = self.rgb_transform(img_d)

        if gt == 0:
            gt = torch.zeros(
                [1, 224, 224])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        return (img, img_d), gt[:1], label, rgb_path


def get_data_loader(split, class_name, dataset_path, img_size = 224, batch_size = 1, shuffle = False):
    if split in ['train']:
        dataset = TrainValDataset(split = "train", class_name = class_name, img_size = img_size, dataset_path = dataset_path)
    elif split in ['validation']:
        dataset = TrainValDataset(split = "validation", class_name = class_name, img_size = img_size, dataset_path = dataset_path)
    elif split in ['test']:
        dataset = TestDataset(class_name = class_name, img_size = img_size, dataset_path = dataset_path)

    data_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle, 
                             num_workers = 1, drop_last = False, pin_memory = True)
    
    return data_loader
