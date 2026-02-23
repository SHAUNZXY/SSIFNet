import os
import json
import random
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from core.utils import (TrainZipReader, TestZipReader, Stack,
                        ToTorchFormatTensor, GroupRandomHorizontalFlip)
import warnings
import copy
warnings.filterwarnings("ignore")

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, debug=False):
        self.args = args
        self.num_local_frames = args['num_local_frames']
        self.num_ref_frames = args['num_ref_frames']
        self.size = self.w, self.h = (args['w'], args['h'])

        json_path = os.path.join(args['data_root'], args['name'], 'train.json')
        with open(json_path, 'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())
        if debug:
            self.video_names = self.video_names[:100]

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def _sample_index(self, length, sample_length, num_ref_frame=3):
        complete_idx_set = list(range(length))
        pivot = random.randint(sample_length+num_ref_frame, length)
        local_idx = complete_idx_set[(pivot-sample_length):pivot]
        
        remain_idx = list(range(pivot-sample_length))
        sample_idx = []
        part_num = len(remain_idx) // num_ref_frame
        for i in range(num_ref_frame):
            remain_idx_part = remain_idx[i*part_num:(i+1)*part_num]
            sample_idx.append(random.sample(remain_idx_part, 1)[0])
        ref_index = sorted(sample_idx)

        return local_idx + ref_index

    def load_item(self, index):
        video_name = self.video_names[index]
        selected_index = self._sample_index(self.video_dict[video_name], self.num_local_frames, self.num_ref_frames)

        # read video frames
        frames_left = []
        frames_right = []
        masks_left = []

        example_mask = np.zeros((self.h,self.w))
        example_mask_ratio = 0.2 + random.random() * 0.4
        example_mask_ratio_iter = 0
        
        while example_mask_ratio_iter < example_mask_ratio:
            example_start_x = np.random.randint(self.h)
            example_start_y = np.random.randint(self.w)
            example_angle = 0.01+np.random.randint(4)
            example_rand_angle = np.random.rand()
            if example_rand_angle < 0.5:
                example_angle = 2 * 3.1415926 - example_angle
            example_length = 120+np.random.randint(60)
            example_brush_w = 80 + np.random.randint(40)
            example_end_x = (example_start_x + example_length * np.sin(example_angle)).astype(np.int32)
            example_end_y = (example_start_y + example_length * np.cos(example_angle)).astype(np.int32)
            
            cv2.line(example_mask, (example_start_y, example_start_x), (example_end_y, example_end_x), 1.0, example_brush_w)
            example_start_x, example_start_y = example_end_x, example_end_y
        
            example_mask_ratio_iter = np.sum(example_mask)/(self.h*self.w)
        
        rand_sample = np.random.rand()
        
        for idx in selected_index:
            video_left_path = os.path.join(self.args['data_root'], self.args['name'], 'JPEGImages', f'{video_name}_left.zip')
            img_left, zfile_1 = TrainZipReader.imread(video_left_path, idx)
            img_left = img_left.convert('RGB')
            img_left = img_left.resize(self.size)
            frames_left.append(img_left)
                
            video_right_path = os.path.join(self.args['data_root'], self.args['name'], 'JPEGImages', f'{video_name}_right.zip')
            img_right, zfile_2 = TrainZipReader.imread(video_right_path, idx)
            img_right = img_right.convert('RGB')
            img_right = img_right.resize(self.size)
            frames_right.append(img_right)
            
            if rand_sample < 0.5:
                
                mask = np.zeros((self.h,self.w))
                mask_ratio = 0.2 + random.random() * 0.4
                mask_ratio_iter = 0
            
                while mask_ratio_iter < mask_ratio:
                    start_x = np.random.randint(self.h)
                    start_y = np.random.randint(self.w)
                    angle = 0.01+np.random.randint(4)
                    rand_angle = np.random.rand()
                    if rand_angle < 0.5:
                        angle = 2 * 3.1415926 - angle
                    length = 80+np.random.randint(80)
                    brush_w = 35 + np.random.randint(10)
                    end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                    end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                    cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                    start_x, start_y = end_x, end_y
                
                    mask_ratio_iter = np.sum(mask)/(self.h*self.w)

                assert(len(np.unique(mask))==2)
                mask_new = Image.fromarray(mask.astype(np.uint8)*255)
                masks_left.append(mask_new)

            else:
                mask = copy.deepcopy(example_mask)
                mask_new = Image.fromarray(mask.astype(np.uint8)*255)
                masks_left.append(mask_new)

        frames = frames_left + frames_right
        frames = GroupRandomHorizontalFlip()(frames)
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        
        masks = masks_left + masks_left
        mask_tensors = self._to_tensors(masks)
        
        return frame_tensors, mask_tensors, video_name



class TestDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.size = (args["w"] , args["h"])
        self.fold = args["fold"]

        with open(os.path.join(args["data_root"], args["dataset"], 'test.json'),
                    'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]
        frame_name_sort = np.sort(os.listdir(os.path.join(self.args["data_root"], self.args["dataset"], self.fold, video_name+'_left')))

        # read video frames
        frames_left = []
        frames_right = []
        masks_left = []
        
        for i, idx in enumerate(frame_name_sort):

            video_left_path = os.path.join(self.args['data_root'], self.args["dataset"], 'JPEGImagesTest', f'{video_name}_left.zip')
            img_left, zfile_1 = TestZipReader.imread(video_left_path, i)
            img_left = img_left.convert('RGB')
            img_left = img_left.resize(self.size)
            frames_left.append(img_left)
                
            video_right_path = os.path.join(self.args['data_root'], self.args["dataset"], 'JPEGImagesTest', f'{video_name}_right.zip')
            img_right, zfile_2 = TestZipReader.imread(video_right_path, i)
            img_right = img_right.convert('RGB')
            img_right = img_right.resize(self.size)
            frames_right.append(img_right)
            
            mask_path = os.path.join(self.args["data_root"], self.args["dataset"], self.fold, video_name+'_left', frame_name_sort[i])
            mask = Image.open(mask_path).resize(self.size, Image.NEAREST).convert('L')
            masks_left.append(mask)

        frames = frames_left + frames_right
        frames_PIL = [np.array(f).astype(np.uint8) for f in frames]
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        
        masks = masks_left + masks_left
        mask_tensors = self._to_tensors(masks)
        
        return frame_tensors, mask_tensors, video_name, frames_PIL
