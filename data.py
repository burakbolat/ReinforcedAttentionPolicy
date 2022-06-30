import os
import copy
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class MiniImageNetDatasetSampler():
    """Create Dataset.
    Dataset size is small. Thus, directly loaded to the memory.
    Unlike, most of the time, the data path is loaded first, then, image is opened at the runtime.
    Images from: https://lyy.mpi-inf.mpg.de/mtl/download/
    """
    def __init__(self, img_dir=None, transform=None, read_all=True):
        self.img_dir = img_dir
        
        self.data_dict = {}
        for root, dirs, _ in os.walk(img_dir):
            for class_ind, class_num in enumerate(dirs):
                class_dir = os.path.join(root, class_num)
                img_arr = []
                for _, _, files in os.walk(class_dir):
                    for ind, file in enumerate(files):
                        img_path = os.path.join(class_dir, file)
                        img = read_image(img_path)/255.0
                        if transform: img = transform(img)
                        img_arr.append(img)
                        if not read_all and ind+1 == 4:
                            break
                self.data_dict[class_ind] = copy.deepcopy(img_arr)
        assert self.data_dict != {}, "Data Dict is empty!"

    def random_subset(self, n_classes):
        random_classes = torch.randint(0, 1)
        pass

if __name__=="__main__":
    minSet = MiniImageNetDatasetSampler("images/train", read_all=False)
