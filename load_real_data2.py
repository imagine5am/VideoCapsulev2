import config
import cv2
import h5py
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
    

def get_annotations(split_type='train'):
    video_anns = {}
    with h5py.File('./realvid_ann.hdf5', 'r') as hf:
        split_grp = hf.get(split_type)
        for video in split_grp.keys():
            video_anns[video] = {}
            video_grp = split_grp.get(video)
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
                
                
def resize_and_pad(im):
    in_h, in_w = im.shape[0], im.shape[1]
    out_h, out_w = config.vid_h, config.vid_w
    
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


def create_mask(pts, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    mask = Image.fromarray(mask, 'L')
    draw = ImageDraw.Draw(mask)
    for pt in pts:
        draw.polygon(pt.tolist(), fill=1)
    del draw
    mask = np.asarray(mask).copy()
    return mask


def load_masked_video(anns, in_shape, n_frames):
    bbox = np.zeros((n_frames, len(config.ann_types), config.vid_h, config.vid_w, 1), dtype=np.uint8)
    for frame_num in range(n_frames):
        for idx, ann_type in enumerate(config.ann_types):
            pts = anns[ann_type][frame_num]
            if pts.size != 0:
                mask = create_mask(pts, in_shape)
                mask_resized = resize_and_pad(mask)
                bbox[frame_num, idx] = np.expand_dims(mask_resized, axis=-1)
    return bbox

                
def get_clips(video, bbox, skip_frames=1, start_rand=True):
    clip_len = 8
            
    # Skip frames after a random start
    if start_rand:
        start_loc = np.random.randint(0, skip_frames)
    else:
        start_loc = 0
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
            
        if np.any(np.sum(bbox, axis=(1, 2, 3)) > 0):
            clips_list.append((clip, bbox))
    
    return clips_list
    
                
class ExternalTestDataLoader():
    def __init__(self):
        print('Running ExternalTestDataLoader...')
        self.test_files = get_annotations('test')
        
        self.videos_left = self.n_videos = len(self.test_files.keys())
        
        self.data_queue = []
        self.load_thread_condition = Condition()
        self.load_thread = Thread(target=self.__load_and_process_data)
        self.load_thread.start()
        
        print('[ExternalTestDataLoader] Waiting 60 (s) to load data')
        time.sleep(60)
            
    
    def __load_and_process_data(self):
        while self.test_files:
            while len(self.data_queue) >= 30:
                with self.load_thread_condition:
                    self.load_thread_condition.wait()
            
            video_name = random.choice(list(self.test_files.keys())) 
            anns = self.test_files.pop(video_name)
            dataset = anns['dataset']
            video_loc = anns['loc']
            n_frames = anns['n_frames'] if dataset!='icdar' else None
            
            if dataset == 'icdar':
                video_orig = skvideo.io.vread(video_loc)
                n_frames, h, w, _ = video_orig.shape
                in_shape = (h, w)
            elif dataset == 'minetto':
                in_shape = (480, 640)
            elif dataset == 'yvt':
                in_shape = (720, 1280)

            video = np.zeros((n_frames, config.vid_h, config.vid_w, 3), dtype=np.uint8)
            for frame_num in range(n_frames):
                if dataset == 'icdar':
                    frame_resized = resize_and_pad(video_orig[frame_num])
                elif dataset == 'minetto':
                    frame_loc = video_loc + '%06d.png' % frame_num
                    frame = cv2.cvtColor(cv2.imread(frame_loc), cv2.COLOR_BGR2RGB)
                    frame_resized = resize_and_pad(frame)
                elif dataset == 'yvt':
                    frame_loc = video_loc+'%d.jpg' % frame_num
                    frame = cv2.cvtColor(cv2.imread(frame_loc), cv2.COLOR_BGR2RGB)
                    frame_resized = resize_and_pad(frame)
            video[frame_num] = frame_resized
                  
            bbox = load_masked_video(anns, in_shape, n_frames)
            label = -1
            self.data_queue.append((video, bbox, label))
            
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
        return self.data_queue != [] or self.test_files != {}


class ExternalTrainDataLoader():
    def __init__(self):
        print('Running ExternalTrainDataLoader...')
        self.train_files = get_annotations()
        self.videos_left = self.n_videos = len(self.train_files.keys())
            
        self.data_queue = []
        self.load_thread_condition = Condition()
        self.load_thread = Thread(target=self.__load_and_process_data)
        self.load_thread.start()
        
        print('[ExternalTrainDataLoader] Waiting 60 (s) to load data')
        time.sleep(60)
       
        
    def __load_and_process_data(self):
        while self.train_files:
            while len(self.data_queue) >= 100:
                with self.load_thread_condition:
                    self.load_thread_condition.wait()
            
            video_name = random.choice(list(self.train_files.keys())) 
            anns = self.train_files.pop(video_name)
            self.videos_left -= 1
            
            dataset = anns['dataset']
            video_loc = anns['loc']
            n_frames = anns['n_frames'] if dataset!='icdar' else None
            
            if dataset == 'icdar':
                video_orig = skvideo.io.vread(video_loc)
                n_frames, h, w, _ = video_orig.shape
                in_shape = (h, w)
            elif dataset == 'minetto':
                in_shape = (480, 640)
            elif dataset == 'yvt':
                in_shape = (720, 1280)

            video = np.zeros((n_frames, config.vid_h, config.vid_w, 3), dtype=np.uint8)
            for frame_num in range(n_frames):
                if dataset == 'icdar':
                    frame_resized = resize_and_pad(video_orig[frame_num])
                elif dataset == 'minetto':
                    frame_loc = video_loc + '%06d.png' % frame_num
                    frame = cv2.cvtColor(cv2.imread(frame_loc), cv2.COLOR_BGR2RGB)
                    frame_resized = resize_and_pad(frame)
                elif dataset == 'yvt':
                    frame_loc = video_loc+'%d.jpg' % frame_num
                    frame = cv2.cvtColor(cv2.imread(frame_loc), cv2.COLOR_BGR2RGB)
                    frame_resized = resize_and_pad(frame)
                
                video[frame_num] = frame_resized
                  
            bbox = load_masked_video(anns, in_shape, n_frames)
            clips_list = get_clips(video, bbox, skip_frames=config.frame_skip)
            self.data_queue.extend(clips_list)
            
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
    while data_gen.has_data:
        video, mask, _ = data_gen.get_next_video()
        for idx, ann_type in enumerate(config.ann_types):
            bbox = mask[:,idx,:,:,:]
            save_masked_video(ann_type[:4]+'/'+str(name), video, bbox)
            name += 1