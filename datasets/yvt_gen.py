import cv2
import os
import numpy as np
import pandas as pd
import random
import time

from threading import Thread, Condition
from PIL import Image, ImageDraw
from skvideo.io import vwrite

base_dir = '../../data/YVT/'
ann_dir = 'annotations/'
frames_dir = 'frames/'
in_h, in_w = 405, 720
out_h, out_w = 256, 480


def save_masked_video(name, video, mask):
    alpha = 0.5
    color = np.zeros((3,)) + [0.0, 0, 1.0]
    masked_vid = np.where(np.tile(mask, [1, 1, 3]) == 1, video * (1 - alpha) + alpha * color, video)
    vwrite(name+'_segmented.avi', (masked_vid * 255).astype(np.uint8))


def resize_and_pad(im):
    if out_w / out_h > in_w / in_h:
        h, w = out_h, in_w * out_h // in_h
    elif out_w / out_h < in_w / in_h:
        h, w =  in_h * out_w // in_w, out_w

    im = cv2.resize(im, (w, h))
    delta_w = out_w - w
    delta_h = out_h - h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return im


def parse_ann(file):
    colnames=['track id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label'] 
    df = pd.read_csv(file, sep=' ', header=None, names=colnames)
    return df


def create_mask(pts):
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask = Image.fromarray(mask, 'L')
    draw = ImageDraw.Draw(mask)
    for pt in pts:
        draw.polygon(pt.tolist(), fill=1)
    del draw
    mask = np.asarray(mask).copy()
    return mask


class YVT_Gen():
    def __init__(self, split_type='train'): # 'train', 'test'
        self.split_type = split_type
        self.n_videos = len(os.listdir(base_dir+frames_dir+self.split_type))
        self.videos_left = self.n_videos
        self.data_queue = []
        
        self.load_thread_condition = Condition()
        self.load_thread = Thread(target=self.__load_and_process_data)
        self.load_thread.start()
        
        print('Running YVTGen...')
        print('Waiting 5 (s) to load data')
        time.sleep(5)
        
        
    def __load_and_process_data(self):
        for name, video, mask in self.get_vid_and_mask():
            if len(self.data_queue) >= 8:
                with self.load_thread_condition:
                    self.load_thread_condition.wait()
            self.data_queue.append((name, video, mask))
        print('[YVTGen] Loading data thread finished')
            
            
    def get_vid_and_mask(self):
        allfiles = os.listdir(base_dir+frames_dir+self.split_type)
        if self.split_type == 'train':
            random.shuffle(allfiles)
        for video_dir in allfiles:
            ann_file = base_dir+ann_dir+self.split_type+'/'+video_dir+'.txt'
            df = parse_ann(ann_file)

            video_dir = base_dir+frames_dir+self.split_type+'/'+video_dir
            n_frames = len(os.listdir(video_dir))
                
            video = np.zeros((n_frames, out_h, out_w, 3), dtype=np.uint8)
            mask = np.zeros((n_frames, out_h, out_w, 1), dtype=np.uint8)
            
            for frame_num in range(n_frames):
                frame_loc = video_dir+'/%d.jpg' % frame_num
                frame = cv2.cvtColor(cv2.imread(frame_loc), cv2.COLOR_BGR2RGB)
                frame_resized = resize_and_pad(frame)
                video[frame_num] = frame_resized
                
                pts = df[(df['frame']==frame_num) & (df['lost']!=1) & (df['occluded']!=1)] \
                        [['xmin','ymin', 'xmax', 'ymin', 'xmax','ymax', 'xmin', 'ymax']].to_numpy()
                if pts.size != 0:
                    frame_mask = create_mask(pts)
                    mask_resized = resize_and_pad(frame_mask)
                    mask[frame_num] = np.expand_dims(mask_resized, axis=-1)  
            # save_masked_video(video_dir[:-4], video/255., mask)
            yield video_dir[:-4], video/255., mask    
            
            
    def get_next_video(self):
        while len(self.data_queue) == 0:
            # print('[YVTGen] Waiting on data')
            time.sleep(5)
        self.videos_left -= 1
        if self.load_thread.is_alive():
            with self.load_thread_condition:
                self.load_thread_condition.notifyAll()
        return self.data_queue.pop(0)


    def has_data(self):
        return self.videos_left > 0


if __name__ == "__main__":
    yvt_gen = YVT_Gen()
    while yvt_gen.has_data():
        name, video, mask = yvt_gen.get_next_video()
        print(name)
        save_masked_video(name, video, mask)