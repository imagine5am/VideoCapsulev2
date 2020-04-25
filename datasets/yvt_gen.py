import cv2
import os
import numpy as np
import pandas as pd
import time

from threading import Thread, Condition
from PIL import Image, ImageDraw
from skvideo.io import vread, vwrite

base_dir = '/home/shivam/Downloads/YVT/'
ann_dir = 'annotations/'
frames_dir = 'frames/'
in_h, in_w = 405, 720
out_h, out_w = 256, 480
split_type = 'train'        # 'train', 'test'


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
        draw.rectangle(pt.tolist(), fill=1)
    del draw
    mask = np.asarray(mask).copy()
    return mask


class YVT_Gen():
    def __init__(self):
        # self.n_videos = 63 if split_type=='train' else 59   #num of tracks
        self.n_videos = len(os.listdir(base_dir+frames_dir+split_type))
        self.videos_left = self.n_videos
        self.data_queue = []
        
        self.load_thread_condition = Condition()
        self.load_thread = Thread(target=self.__load_and_process_data)
        self.load_thread.start()
        
        print('Running YVTGen...')
        print('Waiting 30 (s) to load data')
        time.sleep(30)
        
        
    def __load_and_process_data(self):
        for name, video, mask in self.get_vid_and_mask():
            while len(self.data_queue) >= 10:
                with self.load_thread_condition:
                    self.load_thread_condition.wait()
                # time.sleep(1)
            self.data_queue.append((name, video, mask))
        print('Loading data thread finished')
            
            
    def get_vid_and_mask(self):
        for video_dir in os.listdir(base_dir+frames_dir+split_type):
            ann_file = base_dir+ann_dir+split_type+'/'+video_dir+'.txt'
            df = parse_ann(ann_file)

            base_track_dir = base_dir+frames_dir+split_type+'/'+video_dir+'/0/'
            num_tracks = len(os.listdir(base_track_dir))
            frame_num, n_frames = 0, 0
            for track_num in range(num_tracks):
                n_frames += len(os.listdir(base_track_dir+str(track_num)+'/'))
                
            video = np.zeros((n_frames, out_h, out_w, 3), dtype=np.uint8)
            mask = np.zeros((n_frames, out_h, out_w, 1), dtype=np.uint8)
            
            
            for track_num in range(num_tracks):
                num_track_frames = len(os.listdir(base_track_dir+str(track_num)+'/'))    
                
                for frame_num in range(frame_num, frame_num+num_track_frames):
                    frame_loc = base_track_dir+str(track_num)+'/%d.jpg' % frame_num
                    frame = cv2.cvtColor(cv2.imread(frame_loc), cv2.COLOR_BGR2RGB)
                    frame_resized = resize_and_pad(frame)
                    
                    pts = df[(df['frame']==frame_num) & (df['lost']!=1) & (df['occluded']!=1)] \
                            [['xmin','ymin','xmax','ymax']].to_numpy()
                    frame_mask = create_mask(pts)
                    mask_resized = resize_and_pad(frame_mask)
                    
                    video[frame_num] = frame_resized
                    mask[frame_num] = np.expand_dims(mask_resized, axis=-1)
                    '''
                    cv2.imwrite('mask.jpg', mask_resized)
                    cv2.imwrite('frame.jpg', frame_resized)
                    input()
                    plt.imshow(frame_resized)
                    plt.show()
                    '''    
                frame_num += 1

            save_masked_video(video_dir[:-4], video/255., mask)
            yield video_dir[:-4], video/255., mask    
            
            
    def get_next_video(self):
        while len(self.data_queue) == 0:
            print('Waiting on data')
            time.sleep(5)
        self.videos_left -= 1
        if self.load_thread.is_alive():
            self.load_thread_condition.notifyAll()
        return self.data_queue.pop(0)

    def has_data(self):
        return self.videos_left > 0

if __name__=='__main__':
    yvt_gen = YVT_Gen()
    while yvt_gen.has_data():
        name, _, _ = yvt_gen.get_next_video()
        print(name)