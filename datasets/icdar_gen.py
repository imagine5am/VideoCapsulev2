import cv2
import numpy as np
import os
import skvideo.io  
import time
import xml.etree.ElementTree as ET

from threading import Thread, Condition
from PIL import Image, ImageDraw 

base_dir = '/mnt/data/Rohit/ICDARVideoDataset/text_in_Video/ch3_test/'
# base_dir = '/mnt/data/Rohit/ICDARVideoDataset/text_in_Video/ch3_train/'
out_h, out_w = 128, 240

def resize_and_pad(shape, im):
    in_h, in_w = shape[0], shape[1]
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
    '''
    Returns a dict which is something like:
    {frame_num:{object_id: polygon_pts_list, ...}, ...}
    '''
    tree = ET.parse(file)
    root = tree.getroot()
    ann = {}
    for frame in root.findall('./frame'):
        frame_num = int(frame.attrib['ID']) - 1
        objects = {}
        for object in frame.findall('./object'):
            id = int(object.attrib['ID'])
            pts = []
            for pt in object.findall('./Point'):
                pts.append((int(pt.attrib['x']), int(pt.attrib['y'])))
            objects[id] = pts
        ann[frame_num] = objects
    return ann


def create_mask(shape, pts):
    mask = np.zeros(shape, dtype=np.uint8)
    mask = Image.fromarray(mask, 'L')
    draw = ImageDraw.Draw(mask)
    for pt in pts:
        draw.polygon(pt, fill=1)
    del draw
    mask = np.asarray(mask).copy()
    return mask


def list_vids(dir):
    allfiles = os.listdir(dir)
    files = [ fname for fname in allfiles if fname.endswith('.mp4')]
    return files


class ICDAR_Gen():
    def __init__(self):
        self.n_videos = len(list_vids(base_dir))
        self.videos_left = self.n_videos
        self.data_queue = []
        
        self.load_thread_condition = Condition()
        self.load_thread = Thread(target=self.__load_and_process_data)
        self.load_thread.start()
        
        print('Running ICDARGen...')
        print('Waiting 30 (s) to load data')
        time.sleep(30)
        
    def __load_and_process_data(self):
        for name, video, mask in self.get_vid_and_mask():
            if len(self.data_queue) >= 2:
                with self.load_thread_condition:
                    self.load_thread_condition.wait()
            self.data_queue.append((name, video, mask))
        print('Loading data thread finished')
            
    def get_vid_and_mask(self):
        for video_name in list_vids(base_dir):
            ann_file = base_dir+video_name[:-4]+'_GT.xml'
            ann = parse_ann(ann_file)

            video_orig = skvideo.io.vread(base_dir+video_name)
            n_frames, h, w, ch = video_orig.shape
            video = np.zeros((n_frames, out_h, out_w, 3), dtype=np.uint8)
            mask = np.zeros((n_frames, out_h, out_w, 1), dtype=np.uint8)
            
            for idx in range(n_frames):
                video[idx] = resize_and_pad((h, w), video_orig[idx])
                
                if idx in ann:
                    polygons = ann[idx]
                    if polygons:
                        frame_mask = create_mask((h, w), list(polygons.values()))
                        mask_resized = resize_and_pad((h, w), frame_mask)
                        mask[idx] = np.expand_dims(mask_resized, axis=-1)
            yield video_name[:-4], video/255., mask
            
    def get_next_video(self):
        while len(self.data_queue) == 0:
            print('Waiting on data')
            time.sleep(5)
        self.videos_left -= 1
        if self.load_thread.is_alive():
            with self.load_thread_condition:
                self.load_thread_condition.notifyAll()
        return self.data_queue.pop(0)

    def has_data(self):
        return self.videos_left > 0
