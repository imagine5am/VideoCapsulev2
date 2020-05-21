import config
import cv2
import gc
import h5py
import math
import numpy as np
import random
import skvideo.io
import time

from PIL import Image, ImageDraw
from threading import Thread, Condition

def save_masked_video(name, video, mask):
    alpha = 0.5
    color = np.zeros((3,)) + [0.0, 0, 1.0]
    masked_vid = np.where(np.tile(mask, [1, 1, 3]) == 1, video * (1 - alpha) + alpha * color, video)
    print('Writing', name + '_segmented.avi')
    skvideo.io.vwrite(name+'_segmented.avi', (masked_vid * 255).astype(np.uint8))
    

def get_annotations(split_type='train', dataset='all'):
    video_anns = {}
    with h5py.File('./realvid_ann.hdf5', 'r') as hf:
    # with h5py.File('./realvid13_ann.hdf5', 'r') as hf:
    # with h5py.File('./realvid_ann_neeti.hdf5', 'r') as hf:
        split_grp = hf.get(split_type)
        for video in split_grp.keys():
            video_grp = split_grp.get(video)
            if dataset != 'all' and dataset != video_grp.attrs['dataset']:
                continue
            else:   
                video_anns[video] = {}
                video_anns[video]['dataset'] = video_grp.attrs['dataset']
                video_anns[video]['loc'] = video_grp.attrs['loc']

                if video_anns[video]['dataset'] in ['minetto', 'yvt']:
                    video_anns[video]['n_frames'] = int(video_grp.attrs['n_frames'])

                for ann_type in config.ann_types:
                    ann_grp = video_grp.get(ann_type)
                    ann = {}
                    for frame_num in ann_grp.keys():
                        ann[int(frame_num)] = ann_grp.get(frame_num)[()].astype(np.int32)
                    video_anns[video][ann_type] = ann
    
    return video_anns
                

def resize_and_pad(video_orig, bbox_orig):
    n_frames, in_h, in_w, _ = video_orig.shape
    out_h, out_w = config.vid_h, config.vid_w
    
    if out_w / out_h > in_w / in_h:
        h, w = out_h, in_w * out_h // in_h
    elif out_w / out_h < in_w / in_h:
        h, w =  in_h * out_w // in_w, out_w
        
    delta_w = config.vid_w - w
    delta_h = config.vid_h - h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    video = np.zeros((n_frames, config.vid_h, config.vid_w, 3), dtype=np.uint8)
    bbox = np.zeros((n_frames, len(config.ann_types), config.vid_h, config.vid_w, 1), dtype=np.uint8)
    
    for frame_num in range(n_frames):
        vid_im = cv2.resize(video_orig[frame_num], (w, h))
        video[frame_num] = cv2.copyMakeBorder(vid_im, top, bottom, left, right, cv2.BORDER_CONSTANT)
        
        for idx, ann_type in enumerate(config.ann_types):
            mask = cv2.resize(bbox_orig[frame_num][idx], (w, h))
            mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT)
            bbox[frame_num][idx] = np.expand_dims(mask, axis=-1)
    
    return video, bbox


def create_mask(pts, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    mask = Image.fromarray(mask, 'L')
    draw = ImageDraw.Draw(mask)
    for pt in pts:
        draw.polygon(pt.tolist(), fill=1)
    del draw
    mask = np.asarray(mask).copy()
    return mask


def load_mask(anns, frame_num, m_shape, out_shape=None):
    if out_shape == None:
        multi_mask = np.zeros((len(config.ann_types), m_shape[0], m_shape[1], 1), dtype=np.uint8)
    else:
        multi_mask = np.zeros((len(config.ann_types), out_shape[0], out_shape[1], 1), dtype=np.uint8)
        
    for idx, ann_type in enumerate(config.ann_types):
        if frame_num in anns[ann_type]:
            pts = anns[ann_type][frame_num]
            if pts.size != 0:
                mask = create_mask(pts, m_shape)
                if out_shape != None:
                    mask = cv2.resize(mask, (out_shape[1], out_shape[0]))

                multi_mask[idx] = np.expand_dims(mask, axis=-1)
    return multi_mask


def load_video_and_mask(anns):
    dataset = anns['dataset']
    video_loc = anns['loc']
    n_frames = anns['n_frames'] if dataset!='icdar' else None
    
    if dataset == 'icdar':
        video_orig = skvideo.io.vread(video_loc)
        n_frames, h, w, _ = video_orig.shape
        f_shape = m_shape = (h, w)
    elif dataset == 'minetto':
        f_shape = m_shape = (480, 640)
    elif dataset == 'yvt':
        f_shape = (405, 720)
        m_shape = (720, 1280)

    if dataset != 'icdar':
        video_orig = np.zeros((n_frames, f_shape[0], f_shape[1], 3), dtype=np.uint8)
    bbox_orig = np.zeros((n_frames, len(config.ann_types), f_shape[0], f_shape[1], 1), dtype=np.uint8)
    
    for frame_num in range(n_frames):
        if dataset == 'icdar':
            bbox_orig[frame_num] = load_mask(anns, frame_num, m_shape, out_shape=None)
        
        elif dataset == 'minetto':
            frame_loc = video_loc + '%06d.png' % frame_num
            video_orig[frame_num] = cv2.cvtColor(cv2.imread(frame_loc), cv2.COLOR_BGR2RGB)
            bbox_orig[frame_num] = load_mask(anns, frame_num, m_shape, out_shape=None)
        
        elif dataset == 'yvt':
            frame_loc = video_loc+'%d.jpg' % frame_num
            video_orig[frame_num] = cv2.cvtColor(cv2.imread(frame_loc), cv2.COLOR_BGR2RGB)
            bbox_orig[frame_num] = load_mask(anns, frame_num, m_shape, out_shape=f_shape)
    
    return video_orig, bbox_orig



def random_crop(video_orig, bbox_orig):
    _, in_h, in_w, _ = video_orig.shape
    scale = math.sqrt(random.uniform(0.7, 1))
    out_h, out_w = int(in_h * scale), int(in_w * scale)
    
    ratio = random.uniform(0.7, 1)
    h_or_w = random.choice(['h', 'w'])
    if h_or_w == 'h':
        out_h = int(out_h * ratio)
    else:
        out_w = int(out_w * ratio)
        
    x = random.randint(0, in_w - out_w)
    y = random.randint(0, in_h - out_h)
    video = video_orig[:, y:y+out_h, x:x+out_w,:]
    bbox =  bbox_orig[:, :, y:y+out_h, x:x+out_w,:]
    return video, bbox
           
'''             
def get_clips(video, bbox):
    clip_len = 8
            
    # Skip frames after a random start
    start_loc = random.randint(0,2)
    skip_frames = random.randint(1, 5)
    
    skip_vid = video[start_loc::skip_frames]
    skip_bbox = bbox[start_loc::skip_frames]
    
    # Process the video into 8 frame clips
    n_frames = skip_vid.shape[0]
    clips_list = []
    for clip_start in range(0, n_frames, clip_len):
        clip = skip_vid[clip_start:clip_start+clip_len]
        bbox = skip_bbox[clip_start:clip_start+clip_len]
        
        if clip.shape[0] != clip_len:
            remaining_frames = clip_len - clip.shape[0]
            clip = np.append(clip, np.zeros([remaining_frames] + list(clip.shape[1:])), axis=0)
            bbox = np.append(bbox, np.zeros([remaining_frames] + list(bbox.shape[1:])), axis=0)
        
        #if np.any(np.sum(bbox, axis=(1, 2, 3)) > 0):
        #    clips_list.append((clip, bbox))
        
        clips_list.append((clip, bbox))
    return clips_list
'''
    
def get_clips(video, bbox):
    clip_len = 8
            
    # Random Start
    start_loc = random.randint(0,2)
    video = video[start_loc:]
    bbox = bbox[start_loc:]
    
    # Process the video into 8 frame clips
    n_frames = video.shape[0]
    clips_list = []
    i = 0
    
    while i < n_frames:
        skip_frames = random.randint(1, 5)
        video_clip = video[i::skip_frames][:clip_len]
        bbox_clip = bbox[i::skip_frames][:clip_len]
        
        if video_clip.shape[0] != clip_len:
            remaining_frames = clip_len - video_clip.shape[0]
            video_clip = np.append(video_clip, np.zeros([remaining_frames] + list(video_clip.shape[1:])), axis=0)
            bbox_clip = np.append(bbox_clip, np.zeros([remaining_frames] + list(bbox_clip.shape[1:])), axis=0)
        
        clips_list.append((video_clip, bbox_clip))
        i += (clip_len-1) * skip_frames + 1
        
    return clips_list
                
class ExternalTestDataLoader:
    def __init__(self, data_queue_len=15, dataset='all', sec_to_wait=45):
        print('Running ExternalTestDataLoader...')
        self.test_files = get_annotations('test', dataset=dataset)
        
        self.videos_left = self.n_videos = len(self.test_files.keys())
        
        self.data_queue = []
        self.data_queue_len = data_queue_len
        self.load_thread_condition = Condition()
        self.load_thread = Thread(target=self.__load_and_process_data)
        self.load_thread.start()
        
        print('[ExternalTestDataLoader] Waiting %d (s) to load data' % sec_to_wait)
        time.sleep(sec_to_wait)
            
    
    def __load_and_process_data(self):
        while self.test_files:
            while len(self.data_queue) >= self.data_queue_len:
                with self.load_thread_condition:
                    self.load_thread_condition.wait()
            
            video_name = list(self.test_files.keys())[0] 
            anns = self.test_files[video_name]
            video_orig, bbox_orig = load_video_and_mask(anns)
            video, bbox = resize_and_pad(video_orig, bbox_orig)
            
            del video_orig, bbox_orig
            gc.collect()

            label = -1
            self.data_queue.append((video, bbox, label))
            del self.test_files[video_name]
            
        print('[ExternalTestDataLoader] Data Loading complete...')


    def get_next_video(self):
        while len(self.data_queue) == 0:
            print('[ExternalTestDataLoader] Waiting on data')
            time.sleep(5)
                    
        video, mask, label = self.data_queue.pop(0)
        self.videos_left -= 1
        if self.load_thread.is_alive():
            with self.load_thread_condition:
                self.load_thread_condition.notify_all()
                    
        return video/255., mask, label
        
          
    def has_data(self):
        return self.videos_left > 0


class ExternalTrainDataLoader:
    def __init__(self, data_queue_len=150, dataset='all', sec_to_wait=45):
        print('Running ExternalTrainDataLoader...')
        self.train_files = get_annotations()
        self.video_order = list(self.train_files.keys())
        random.shuffle(self.video_order)
        
        self.videos_left = self.n_videos = len(self.train_files.keys())
            
        self.data_queue = []
        self.data_queue_len = data_queue_len
        self.load_thread_condition = Condition()
        self.load_thread = Thread(target=self.__load_and_process_data)
        self.load_thread.start()
        
        print('[ExternalTrainDataLoader] Waiting %d (s) to load data' % sec_to_wait)
        time.sleep(sec_to_wait)
       
        
    def __load_and_process_data(self):
        while self.train_files:
            while len(self.data_queue) >= self.data_queue_len:
                with self.load_thread_condition:
                    self.load_thread_condition.wait()
            
            video_name = self.video_order.pop(0)
            anns = self.train_files[video_name]
            video_orig, bbox_orig = load_video_and_mask(anns)
            video_crop, bbox_crop = random_crop(video_orig, bbox_orig)
            video, bbox = resize_and_pad(video_crop, bbox_crop)

            del video_orig, bbox_orig, video_crop, bbox_crop
            gc.collect()
            
            clips_list = get_clips(video, bbox)
            self.data_queue.extend(clips_list)
            del self.train_files[video_name]
            self.videos_left -= 1
            
        print('[ExternalTrainDataLoader] Data Loading complete...')
        
        
    def get_batch(self, batch_size=5):
        while len(self.data_queue) == 0:
            print('[ExternalTrainDataLoader] Waiting on data')
            time.sleep(5)
        
        batch_size = min(batch_size, len(self.data_queue))
        batch_x, batch_bbox, batch_y = [], [], []
        for _ in range(batch_size):
            vid, bbox = self.data_queue.pop(random.randrange(len(self.data_queue)))
            batch_x.append(vid/255.)
            batch_bbox.append(bbox)
            batch_y.append(-1)
        
        if self.load_thread.is_alive():
            with self.load_thread_condition:
                self.load_thread_condition.notify_all()
                    
        return batch_x, batch_bbox, batch_y
           
            
    def has_data(self):
        return self.data_queue != [] or self.train_files != {}
   
    
if __name__ == "__main__":
    data_gen = ExternalTestDataLoader()
    name = 0
    while data_gen.has_data():
        video, mask, _ = data_gen.get_next_video()
        for idx, ann_type in enumerate(config.ann_types):
            bbox = mask[:,idx,:,:,:]
            save_masked_video(ann_type[:4]+'/'+str(name), video, bbox)
        name += 1