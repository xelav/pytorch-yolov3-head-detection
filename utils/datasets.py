import xml.etree.ElementTree as ET
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from PIL import Image
import random
import logging
import pickle
from utils.plot import *
from utils.utils import xyxy_to_xywh
from utils.augment import ImgAugTransform
from tqdm import tqdm
import glob
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def pad_to_square(img, pad_value=0):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class MultiscaleConcatDataset(ConcatDataset):
    """
    Awkward wrapper to make the collate_fn method as some sort
    of callback for multiscale datasets. Works just like VOCDetection.collate_fn
    """

    def __init__(self, datasets, img_size=416, multiscale=True):
        super(MultiscaleConcatDataset, self).__init__(datasets)

        self.img_size = img_size

        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.multiscale = multiscale
        self.multiscale_interval = 10

        self.batch_count = 0

    def pick_new_img_size(self):

        if self.multiscale:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
            logger.debug(f"changing img_size. New img_size: {self.img_size}")

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))

        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)

        if self.multiscale and self.batch_count % self.multiscale_interval == 0:
            self.pick_new_img_size()
            for dataset in self.datasets:
                dataset.img_size = self.img_size

        self.batch_count += 1

        imgs = torch.stack(imgs)
        return imgs, targets


class VOCDetection(Dataset):
    """
    Dataset class for VOCPascal-like datasets.
    Implements collate_fn method for multiscale training.
    Also makes a cache files for all parsed annotation files.
    """

    def __init__(self, img_dir, annotation_dir, split_file=None, cache_dir=None, img_size=416, filter_labels=None, multiscale=True, augment=False):

        self.img_dir = img_dir
        self.annotation_dir = annotation_dir

        self.file_names = None
        if split_file:
            if os.path.isfile(split_file):
                with open(os.path.join(split_file), "r") as f:
                    self.file_names = [x.strip() for x in f.readlines()]
                print(f"Size of the split: {len(self.file_names)}")
            else:
                split_file=None
                print("Split file not found. Loading whole dataset instead")

        self.filter_labels = filter_labels

        self.img_size = img_size

        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.multiscale = multiscale
        self.multiscale_interval = 10

        self.batch_count = 0

        set_name = os.path.basename(os.path.splitext(split_file)[0]) if split_file else "all"
        self.cache_file = os.path.join(cache_dir, f'annotation_cache_for_{set_name}')
        if cache_dir and os.path.isfile(self.cache_file):
            logging.info(f"loading cached annotation file {set_name}")
            with open(self.cache_file, 'rb') as pickle_file:
                self.objects, self.labels = pickle.load(pickle_file)
        else:
            logging.info(f"Cache file {self.cache_file} couldn't be found")
            logging.info("Parsing annotations...")
            self.objects, self.labels = self.parse_annotation(self.annotation_dir,
                                                              self.img_dir,
                                                              self.filter_labels,
                                                              filenames=self.file_names)
            logging.info(f"Saving cache file {set_name}")
            with open(self.cache_file, 'wb') as pickle_file:
                pickle.dump([self.objects, self.labels], pickle_file)

        aug_pipe = None
        if augment:
            aug_pipe = ImgAugTransform()

        self.aug_pipe = aug_pipe

    @staticmethod
    def parse_annotation(ann_dir, img_dir, labels=None, filenames=None):
        all_imgs = []
        seen_labels = {}

        if filenames is not None:
            annotations = (f + ".xml" for f in filenames)
        else:
            annotations = glob.iglob( os.path.join(ann_dir, '*.xml') ) # TODO: test

        for ann in tqdm(annotations):
            img = {'object': []}

            tree = ET.parse(os.path.join(ann_dir, ann))

            for elem in tree.iter():
                if 'filename' in elem.tag:
                    img['filename'] = elem.text
                if 'width' in elem.tag:
                    img['width'] = int(elem.text)
                if 'height' in elem.tag:
                    img['height'] = int(elem.text)
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}

                    for attr in list(elem):
                        if 'name' in attr.tag:
                            obj['name'] = attr.text

                            if obj['name'] in seen_labels:
                                seen_labels[obj['name']] += 1
                            else:
                                seen_labels[obj['name']] = 1

                            if labels is not None and obj['name'] not in labels:
                                break
                            else:
                                img['object'] += [obj]

                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(round(float(dim.text)))

            if len(img['object']) > 0:
                all_imgs += [img]

        return all_imgs, seen_labels

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        image - FloatTensor(3, IMG_SIZE, IMG_SIZE)
        bbs - FloatTensor(BoundingBoxes, 4) - center_x, center_y, w, h normalized by image size
        """
        sample = self.objects[idx]
        image = np.array(Image.open(
            os.path.join(self.img_dir, sample['filename'])
        ).convert('RGB'))

        assert len(image.shape) == 3

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=t["xmin"], y1=t["ymin"], x2=t["xmax"], y2=t["ymax"]) for t in sample["object"]],
            shape=image.shape
        )

        if self.aug_pipe:
            image, bbs = self.aug_pipe(image, bbs)

        bbs = bbs.to_xyxy_array()
        image = image.transpose(2,0,1)  # (H, W, C) -> (C, H, W)

        image = torch.from_numpy(image.copy()).float()  # .copy to solve the negative stride issue in Pytorch

        # make padded square image
        padded_image, pad = pad_to_square(image, 0)
        _, padded_h, padded_w = padded_image.shape
        bbs[:, [0, 2]] += [pad[0]] * 2
        bbs[:, [1, 3]] += [pad[2]] * 2

        bbs = xyxy_to_xywh(bbs, padded_w, padded_h)
        bbs[:, :2] = bbs[:, :2].clip(0, 1)  # clip out of image centers

        targets = torch.zeros((len(bbs), 6))
        targets[:, 2:] = torch.from_numpy(bbs)

        padded_image = resize(padded_image, (self.img_size, self.img_size))

        return padded_image, targets

    def pick_new_img_size(self):

        if self.multiscale:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
            logger.debug(f"changing img_size. New img_size: {self.img_size}")

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))

        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)

        # Selects new image size every tenth batch
        if self.batch_count % self.multiscale_interval == 0:
            self.pick_new_img_size()

        self.batch_count += 1

        # auf pipe time logging
        # if self.aug_pipe:
        #     logging.info(f"aug time: {self.aug_pipe.time_consumed:.4f} seconds")
        #     self.aug_pipe.time_consumed = 0

        imgs = torch.stack(imgs)
        return imgs, targets


class ScutHeadDataset(VOCDetection):
    """
    Dataset class for SCUT-HEAD
    follows VOC Pascal
    https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release
    """
