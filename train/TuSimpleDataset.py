import torch
from torchvision import transforms, datasets
import cv2
import numpy as np
import json
import os
from torch.utils.data import Dataset
import torch.nn as nn
from torch import optim as optim

# import sys
# # sys.path.append("..")
# # from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# # sam_checkpoint = "sam_vit_h_4b8939.pth"
# # model_type = "vit_h"

# # device = "cuda"

# # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# # sam.to(device=device)

# # mask_generator = SamAutomaticMaskGenerator(sam)


# train_img_size = 128

# class TuSimpleDataset(Dataset):
#     def __init__(self, image_dir, label_files, annotation_files=[], transform=None, mask_transform=None):
#         self.image_dir = image_dir
#         if transform is None:
#             self.transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize((train_img_size, train_img_size)),  # Resize the image
#                 transforms.ToTensor(),  # Convert to tensor
#                 # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (optional)
#             ])
#         else:
#             self.transform = transform
#         if mask_transform is None:
#             self.mask_transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize((train_img_size, train_img_size)),
#                 transforms.ToTensor()  # Only resizing and converting to tensor, no normalization
#             ])
#         else:
#             self.mask_transform = mask_transform
#         self.labels = []
#         for label_file in label_files:
#             self.labels+=self._load_labels(label_file)
#         self.annotation_dict = {}
#         for annotation_file in annotation_files:
#             print(annotation_file)
#             self.annotation_dict = {**self.create_annotation_dict(annotation_file), **self.annotation_dict}

#     def create_annotation_dict(self, path_to_annotation):
#         with open(path_to_annotation) as f:
#             annotation_dict = json.load(f)
#         return annotation_dict

#     def _load_labels(self, label_file):
#         labels = []
#         with open(label_file) as f:
#             for line in f:
#                 obj_loaded = json.loads(line)
#                 if obj_loaded['raw_file'][:13] == 'clips/0313-2/' or obj_loaded['raw_file'][:13] == 'clips/0313-1/'  or obj_loaded['raw_file'][:11] == 'clips/0601/':
#                     labels.append(obj_loaded)
#         return labels

#     def _create_mask(self, label, img_height, img_width):
#         h_samples = label['h_samples']
#         lanes = label['lanes']
#         lane_matrix = np.zeros((img_height, img_width), dtype=np.uint8)

#         def mark_point(matrix, y, x, point_s=3):
#             for i in range(-1*point_s+1, point_s):
#                 for j in range(-1*point_s+1, point_s):
#                     if 0 <= y + i < img_height and 0 <= x + j < img_width:
#                         matrix[y + i, x + j] = 255

#         for lane in lanes:
#             for h, x in zip(h_samples, lane):
#                 if x != -2:
#                     mark_point(lane_matrix, h, x)  # Note: x and h are flipped here for image coordinates

#         return lane_matrix

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         label = self.labels[idx]
#         img_path = os.path.join(self.image_dir, label['raw_file'])
#         try:
#             category = self.annotation_dict['./train_set/'+label['raw_file']]
#         except:
#             category = 0
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         img_height, img_width = image.shape[:2]
#         mask = self._create_mask(label, img_height, img_width)

#         image = self.transform(image)
#         mask = self.mask_transform(mask)  # Apply mask transformation

#         return image, mask, category
# import torch
# from torchvision import transforms, datasets
# import numpy as np
# import json
# import os
# from torch.utils.data import Dataset
# import torch.nn as nn
# from torch import optim as optim

# import sys
# sys.path.append("..")
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# # from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# sam_checkpoint = '/Users/wangtianhe/Desktop/USC/Fall 2023/EE641 Deep Learning Systems/LaneGeneration/src/saved_models/SAM/mobile_sam.pt'
# model_type = "vit_t"

# device_msk = "cpu"

# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device_msk)
# sam.eval()

# mask_generator = SamAutomaticMaskGenerator(sam)


train_img_size = 128
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    # Initialize a white image
    img_shape = sorted_anns[0]['segmentation'].shape
    img = np.ones((img_shape[0], img_shape[1], 3))  # 3-channel white image

    for ann in sorted_anns:
        m = ann['segmentation']
        img[m] = [0, 0, 0]  # Set mask area to black

    return img


class TuSimpleDataset(Dataset):
    def __init__(self, image_dir, label_files, annotation_files=[], transform=None, mask_transform=None, test_mode=False):
        self.test_mode = test_mode
        self.image_dir = image_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((train_img_size, train_img_size)),  # Resize the image
                transforms.ToTensor(),  # Convert to tensor
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (optional)
            ])
        else:
            self.transform = transform
        if mask_transform is None:
            self.mask_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((train_img_size, train_img_size)),
                transforms.ToTensor()  # Only resizing and converting to tensor, no normalization
            ])
        else:
            self.mask_transform = mask_transform
        self.labels = []
        for label_file in label_files:
            self.labels+=self._load_labels(label_file)
        self.annotation_dict = {}
        for annotation_file in annotation_files:
            print(annotation_file)
            self.annotation_dict = {**self.create_annotation_dict(annotation_file), **self.annotation_dict}

    def create_annotation_dict(self, path_to_annotation):
        with open(path_to_annotation) as f:
            annotation_dict = json.load(f)
        return annotation_dict

    def _load_labels(self, label_file):
        labels = []
        with open(label_file) as f:
            for line in f:
                obj_loaded = json.loads(line)
                raw_file_path = obj_loaded['raw_file']
                mask_sam_subdir_path = 'mask_sam/'+raw_file_path[6:]
                
                if os.path.exists(mask_sam_subdir_path):
                    print(mask_sam_subdir_path)
                # if obj_loaded['raw_file'][:13] == 'clips/0313-2/' or obj_loaded['raw_file'][:13] == 'clips/0313-1/'  or obj_loaded['raw_file'][:11] == 'clips/0601/':
                    labels.append(obj_loaded)
        return labels

    def _create_mask(self, label, img_height, img_width):
        h_samples = label['h_samples']
        lanes = label['lanes']
        lane_matrix = np.zeros((img_height, img_width), dtype=np.uint8)

        def mark_point(matrix, y, x, point_s=3):
            for i in range(-1*point_s+1, point_s):
                for j in range(-1*point_s+1, point_s):
                    if 0 <= y + i < img_height and 0 <= x + j < img_width:
                        matrix[y + i, x + j] = 255

        for lane in lanes:
            for h, x in zip(h_samples, lane):
                if x != -2:
                    mark_point(lane_matrix, h, x)  # Note: x and h are flipped here for image coordinates

        return lane_matrix

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not self.test_mode:
            label = self.labels[idx]
            img_path = os.path.join(self.image_dir, label['raw_file'])
            mask_path = 'mask_sam/'+label['raw_file'][6:]
            try:
                category = self.annotation_dict['./train_set/'+label['raw_file']]
            except:
                category = 0
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # mask = mask_generator.generate(image)
            # msk = show_anns(mask)
            # msk = self.transform(msk.astype(np.uint8)*255)
            mask = cv2.imread(mask_path)
            img_height, img_width = image.shape[:2]
            msk_lane = self._create_mask(label, img_height, img_width)
            msk_lane = np.stack((msk_lane,)*3, axis=-1)
            mask = cv2.addWeighted(mask, 1, msk_lane, 1, 0)

            mask = self.mask_transform(mask)  # Apply mask transformation
            image = self.transform(image)
            return image, mask, category
        else:
            label = self.labels[idx]
            img_path = os.path.join(self.image_dir, label['raw_file'])
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
