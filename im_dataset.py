import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms.v2.functional import resize, crop

class CaptionImageNetDataset(Dataset):

    def __init__(self, imagenet_path: str, im_size: int = 256):
        self.imagenet_path = imagenet_path
        self.im_size = im_size
        # Login using e.g. `huggingface-cli login` to access this dataset
        self.ds = load_dataset("Lucasdegeorge/ImageNet_TA_IA", split="train").with_format("torch")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        path = sample['image_path']
        class_path = path.split('_')[0]

        img = decode_image(self.imagenet_path+"/"+class_path+"/"+path)
        c, h, w = img.shape
        if c < 3:
            img = img.tile(3, 1, 1)
        elif c > 3:
            img = img[0:3, :, :]
        sample['size'] = torch.tensor([h, w])

        if min(h,w)<self.im_size:
            #crop
            if h < w :
                nh = self.im_size
                nw = int(self.im_size*w/h)
            else:
                nw = self.im_size
                nh = int(self.im_size*h/w)
            img = resize(img, (nh, nw))
        else:
            nh = h
            nw = w

        # crop
        i = torch.randint(low=0, high=nh-self.im_size, size=(1,)) if nh>self.im_size else torch.zeros(1, dtype=torch.long)
        j = torch.randint(low=0, high=nw-self.im_size, size=(1,)) if nw>self.im_size else torch.zeros(1, dtype=torch.long)
        img = img[:, i:i+self.im_size, j:j+self.im_size]
        sample['crop'] = torch.cat([i/nh, (i+self.im_size)/nh, j/nw,(j+self.im_size)/nw]).round(decimals=2)
        sample['image'] = img/127.5 - 1.0

        return sample


