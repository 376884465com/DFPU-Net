import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input
from utils import read_tif_toTensor


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        
        readtif = read_tif_toTensor.ReadTiff()
        jpg = readtif.readTif_to_Ndarray(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".tif"))
       
        png         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
        
        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [0,1,2]) 
        png         = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

def unet_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = np.array(images)
    pngs        = np.array(pngs)
    seg_labels  = np.array(seg_labels)
    return images, pngs, seg_labels
