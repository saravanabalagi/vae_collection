from torch.utils.data import Dataset
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from random import shuffle
import pickle
import glob
import os


class OxfordRobotcar(Dataset):
    def __init__(self, root, split="train", transform=None, 
                 cache=True, *args, **kwargs) -> None:
        self.root = root
        self.base_folder = 'oxford_robotcar_pickles_v3'
        self.transform = transform
        self.cache = cache
        self.split = split

        filenames_pickle = os.path.join(self.root, self.base_folder, f'filenames_split.pickle')
        imgs_pickle = os.path.join(self.root, self.base_folder, f'data_{split}.pickle')

        if os.path.exists(filenames_pickle) and os.path.exists(imgs_pickle):
            with open(filenames_pickle, 'rb') as f:
                self.filenames = pickle.load(f)[split]
            with open(imgs_pickle, 'rb') as f:
                self.imgs = pickle.load(f)

        else:
            filenames = glob.glob(os.path.join(self.root, self.base_folder) + '/**/*.jpg')        
            shuffle(filenames)

            start, end = self._split(len(filenames))
            self.filenames = filenames[start:end]

            if self.cache:
                self.imgs = []
                for f in tqdm(self.filenames):
                    img = Image.open(f)
                    img = self.pretransform(img)
                    self.imgs.append(img)

        super().__init__()

    def _split(self, num_files):
        start = 0
        end = num_files
        split = self.split

        if split == 'train':
            end = end * 0.6
        elif split == 'valid':
            start = 0.6 * end
            end = 0.8 * end
        elif split == 'test': 
            start = 0.8 * end
        else:
            raise IOError(f'Could not find {split} split')
        return int(start), int(end)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if not self.cache:
            X = Image.open(self.filenames[idx])
            X = self.pretransform(X)
        else: 
            X = self.imgs[idx]
        if self.transform is not None:
            X = self.transform(X)
        return X, 0

    def pretransform(self, X):
        X = transforms.functional.crop(X, 0, 0, 800, 1280)  # crop hood
        X = transforms.functional.resize(X, size=140)
        return X
