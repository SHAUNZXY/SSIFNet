# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
matplotlib.use('agg')
from core.utils import to_tensors
import random 

parser = argparse.ArgumentParser(description="SSIFNet")
parser.add_argument("-v", "--video", type=str, required=True)
parser.add_argument("-c", "--ckpt", type=str, required=True)
parser.add_argument("-m", "--mask", type=str, required=True)
parser.add_argument("--model", type=str, choices=['ssifnet'])
parser.add_argument("--num_ref", type=int, default=5)
parser.add_argument("--neighbor_stride", type=int, default=10)
parser.add_argument("--savefps", type=int, default=5)
parser.add_argument("--dilate", type=int, default=15)
parser.add_argument("--set_size", action='store_true', default=False)
parser.add_argument("--width", type=int)
parser.add_argument("--height", type=int)

args = parser.parse_args()

ref_frame_num = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps


def get_key_ref_index(f, neighbor_ids, ref_frame_num, masks=None):
    if ref_frame_num <= 0:
        return []
    
    ref_frames = sorted(set(range(f + 1)) - set(neighbor_ids))

    if len(ref_frames) <= ref_frame_num:
        return ref_frames

    ref_index = random.sample(ref_frames, ref_frame_num)

    ref_index.sort()
    return ref_index


# read frame-wise masks
def read_mask(mpath, size):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for mp in mnames:
        m = Image.open(os.path.join(mpath, mp))
        m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m,
                        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                        iterations=args.dilate)
        masks.append(Image.fromarray(m * 255))
    return masks


#  read frames from video
def read_frame_from_videos(args):
    vname = args.video
    frames_left = []
    frames_right = []
    if args.use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        count = 0
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames_left.append(image)
            success, image = vidcap.read()
            count += 1

        vname_right = vname.replace("left", "right")
        vidcap = cv2.VideoCapture(vname_right)
        success, image = vidcap.read()
        count = 0
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames_right.append(image)
            success, image = vidcap.read()
            count += 1            

    else:
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname + '/' + name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames_left.append(image)
    return frames_left, frames_right


# resize frames
def resize_frames(frames, size=None):
    if size is not None:
        frames = [f.resize(size) for f in frames]
    else:
        size = frames[0].size
    return frames, size


def main_worker():
    # set up models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model is not None:
        size = (432, 240)
    else:
        size = None

    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    print(f'Loading model from: {args.ckpt}')
    model.eval()

    # prepare datset
    args.use_mp4 = True if args.video.endswith('.mp4') else False
    print(
        f'Loading videos and masks from: {args.video} | INPUT MP4 format: {args.use_mp4}'
    )
    frames_left, frames_right = read_frame_from_videos(args)
    
    frames_left, size = resize_frames(frames_left, size)
    frames_right, size = resize_frames(frames_right, size)
    h, w = size[1], size[0]
    video_length = len(frames_left)
    
    imgs_left = to_tensors()(frames_left).unsqueeze(0) * 2 - 1
    imgs_right = to_tensors()(frames_right).unsqueeze(0) * 2 - 1    
    
    frames_left = [np.array(f).astype(np.uint8) for f in frames_left]
    frames_right = [np.array(f).astype(np.uint8) for f in frames_right]

    masks_left = read_mask(args.mask, size)
    masks_right = read_mask(args.mask.replace("left", "right"), size)
    
    binary_masks_left = [ np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks_left ]
    binary_masks_right = [ np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks_right ]    
    
    masks_left = to_tensors()(masks_left).unsqueeze(0)
    masks_right = to_tensors()(masks_right).unsqueeze(0)
    
    imgs_left, imgs_right, masks_left, masks_right = imgs_left.to(device), imgs_right.to(device), masks_left.to(device), masks_right.to(device)
    comp_frames_left = [None] * video_length
    comp_frames_right = [None] * video_length

    # completing holes by e2fgvi
    print(f'Start test...')
    
    for f in range(neighbor_stride, video_length):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                            min(video_length, f + neighbor_stride + 1))
        ]
        # print("neighbor_ids: ", neighbor_ids)
        ref_ids = get_key_ref_index(f, neighbor_ids, ref_frame_num, masks_left)
        # print("ref_ids", ref_ids)
        selected_imgs_left = imgs_left[:1, neighbor_ids + ref_ids, :, :, :]
        selected_masks_left = masks_left[:1, neighbor_ids + ref_ids, :, :, :]
        selected_imgs_right = imgs_right[:1, neighbor_ids + ref_ids, :, :, :]
        selected_masks_right = masks_right[:1, neighbor_ids + ref_ids, :, :, :]        

        selected_imgs = torch.cat((selected_imgs_left,selected_imgs_right), dim=1)
        selected_masks = torch.cat((selected_masks_left,selected_masks_right), dim=1)
        
        
        with torch.no_grad():
            masked_imgs = selected_imgs * (1 - selected_masks)
            mod_size_h = 60
            mod_size_w = 108
            h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
            w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
            
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [3])],
                3)[:, :, :, :h + h_pad, :]
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [4])],
                4)[:, :, :, :, :w + w_pad]
            
            pred_imgs, pred_flows, pred_disps = model(masked_imgs, len(neighbor_ids))
            pred_imgs = pred_imgs[:, :, :h, :w]
            pred_imgs = (pred_imgs + 1) / 2
            pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
            
            # left
            pred_img_left = pred_imgs[:int(pred_imgs.shape[0]//2),:,:,:]
                
            if f != neighbor_stride:
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    if idx == f:
                        img = np.array(pred_img_left[i]).astype(np.uint8) * binary_masks_left[idx] \
                            + frames_left[idx] * (1 - binary_masks_left[idx])
                        comp_frames_left[idx] = img
            else:
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img_left[i]).astype(np.uint8) * binary_masks_left[idx] \
                        + frames_left[idx] * (1 - binary_masks_left[idx])
                    comp_frames_left[idx] = img
                    
                    
            # right
            pred_img_right = pred_imgs[int(pred_imgs.shape[0]//2):,:,:,:]
                
            if f != neighbor_stride:
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    if idx == f:
                        img = np.array(pred_img_right[i]).astype(np.uint8) * binary_masks_right[idx] \
                            + frames_right[idx] * (1 - binary_masks_right[idx])
                        comp_frames_right[idx] = img
            else:
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img_right[i]).astype(np.uint8) * binary_masks_right[idx] \
                        + frames_right[idx] * (1 - binary_masks_right[idx])
                    comp_frames_right[idx] = img
    
    
    video_name = args.video.split('/')[-1].split('.')[0]
    save_dir_name = 'results' 
    save_inpainting_type = 'inpainting_instrument'
    save_inpainting_fold = os.path.join(save_dir_name, save_inpainting_type)
    save_inpainting_video = save_inpainting_fold + '/' + video_name
    save_inpainting_video_left = os.path.join(save_inpainting_video, 'left')
    save_inpainting_video_right = os.path.join(save_inpainting_video, 'right')

    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)   
    if not os.path.exists(save_inpainting_fold):
        os.makedirs(save_inpainting_fold)    
    if not os.path.exists(save_inpainting_fold):
        os.makedirs(save_inpainting_fold)    
    if not os.path.exists(save_inpainting_video_left):
        os.makedirs(save_inpainting_video_left)    
    if not os.path.exists(save_inpainting_video_right):
        os.makedirs(save_inpainting_video_right)      
    
    for frame_id in range(len(comp_frames_left)):
        inp_left_array = comp_frames_left[frame_id] 
        cv2.imwrite(save_inpainting_video_left + "/" + str(frame_id) + '.jpg', inp_left_array[:,:,::-1])
    for frame_id in range(len(comp_frames_right)):
        inp_right_array = comp_frames_right[frame_id]
        cv2.imwrite(save_inpainting_video_right + "/" + str(frame_id) + '.jpg', inp_right_array[:,:,::-1])       

    print('Saving left videos...')
    save_dir_name = save_inpainting_fold
    
    ext_name = '_left_results.mp4'
    save_base_name = args.video.split('/')[-1]
    save_name = save_base_name.replace(
        '_left.mp4', ext_name) if args.use_mp4 else save_base_name + ext_name
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)
    save_path = os.path.join(save_dir_name, save_name)
    size_new = (size[0]*3, size[1])
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                            default_fps, size_new)
    ori_frames = (((imgs_left[0,...].cpu().permute(0, 2, 3, 1).numpy() + 1)/2)*255).astype(np.uint8)
    ori_masks = (masks_left[0,...].cpu().permute(0, 2, 3, 1).numpy()).astype(np.uint8)
    masked_frames = ori_frames * (1 - ori_masks)
    for f in range(video_length):
        save_comp = comp_frames_left[f].astype(np.uint8)
        save_mask = masked_frames[f,:,:,:].astype(np.uint8)
        save_img = ori_frames[f,:,:,:].astype(np.uint8)
        save_new = np.zeros((save_comp.shape[0], save_comp.shape[1]*3, save_comp.shape[2])).astype(np.uint8)
        save_new[:,:save_comp.shape[1],:] = save_mask
        save_new[:,save_comp.shape[1]:save_comp.shape[1]*2,:] = save_img
        save_new[:,save_comp.shape[1]*2:save_comp.shape[1]*3,:] = save_comp
        writer.write(cv2.cvtColor(save_new, cv2.COLOR_BGR2RGB))
    writer.release()
    print(f'Finish test! The result video is saved in: {save_path}.')


    print('Saving right videos...')
    save_dir_name = save_inpainting_fold
    ext_name = '_right_results.mp4'
    save_base_name = args.video.split('/')[-1]
    save_name = save_base_name.replace(
        '_left.mp4', ext_name) if args.use_mp4 else save_base_name + ext_name
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)
    save_path = os.path.join(save_dir_name, save_name)
    size_new = (size[0]*3, size[1])
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                            default_fps, size_new)
    ori_frames = (((imgs_right[0,...].cpu().permute(0, 2, 3, 1).numpy() + 1)/2)*255).astype(np.uint8)
    ori_masks = (masks_right[0,...].cpu().permute(0, 2, 3, 1).numpy()).astype(np.uint8)
    masked_frames = ori_frames * (1 - ori_masks)
    for f in range(video_length):
        save_comp = comp_frames_right[f].astype(np.uint8)
        save_mask = masked_frames[f,:,:,:].astype(np.uint8)
        save_img = ori_frames[f,:,:,:].astype(np.uint8)
        save_new = np.zeros((save_comp.shape[0], save_comp.shape[1]*3, save_comp.shape[2])).astype(np.uint8)
        save_new[:,:save_comp.shape[1],:] = save_mask
        save_new[:,save_comp.shape[1]:save_comp.shape[1]*2,:] = save_img
        save_new[:,save_comp.shape[1]*2:save_comp.shape[1]*3,:] = save_comp
        writer.write(cv2.cvtColor(save_new, cv2.COLOR_BGR2RGB))
    writer.release()
    print(f'Finish test! The result video is saved in: {save_path}.')

if __name__ == '__main__':
    main_worker()
