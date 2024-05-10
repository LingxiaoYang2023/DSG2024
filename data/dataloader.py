from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import os
import torch

__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader




@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None, load_inputs=False, task_name='', device='cuda:7'):
        super().__init__(root, transforms)
        self.load_inputs = load_inputs
        self.device = device
        self.fpaths = sorted(glob(root + '/*.png', recursive=False))
        # self.fpaths = sorted(glob(root + '/*.jpg', recursive=False))
        # self.fpaths = sorted(glob(root + '/*.JPEG', recursive=False))

        if load_inputs:
            root = os.path.join(root, task_name, "input")
            # self.fpaths = sorted(glob(root + '/*.png', recursive=False))
            print(f'root: {root}')
            # print(f'self.fpaths: {self.fpaths}')
            self.inputs = sorted(glob(root + '/*.pt', recursive=False))

        # self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        # self.fpaths = sorted(glob(root + '/**/*.JPEG', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)

        if self.load_inputs:
            input_path = self.inputs[index]
            input = torch.load(input_path,map_location='cpu')
            # print(f'input_path:{input_path} input:{input.shape}')
            return (img, input)
        
        # return img, fpath

        return img