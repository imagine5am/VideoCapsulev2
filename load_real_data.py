import config
import numpy as np
import random
import time
from datasets.icdar_gen import ICDAR_Gen
from datasets.minetto_gen import Minetto_Gen
from datasets.yvt_gen import YVT_Gen
from threading import Thread, Condition


class ExternalTestDataLoader():
    def __init__(self):
        print('Running ExternalTestDataLoader...')
        self.icdar_gen = ICDAR_Gen(split_type='test')
        self.yvt_gen = YVT_Gen(split_type='test')
        self.minetto_gen = Minetto_Gen()
        
        self.n_videos = self.videos_left = self.icdar_gen.n_videos + self.yvt_gen.n_videos + self.minetto_gen.n_videos
            
    
    def get_next_video(self, get_name=False):
        self.videos_left -= 1
        # for data_gen in [self.icdar_gen, self.yvt_gen, self.minetto_gen]:
        for data_gen in [self.minetto_gen, self.icdar_gen, self.yvt_gen]:
            video_name, video, mask = data_gen.get_next_video()
            mask = np.tile(np.expand_dims(mask, axis=1), [1, 4, 1, 1, 1])
            
            if data_gen.has_data():
                if get_name:
                    return video_name, video, mask
                else:
                    return video, mask
            
            
    def has_data(self):
        return self.videos_left > 0


class ExternalTrainDataLoader():
    def __init__(self):
        print('Running ExternalTrainDataLoader...')
        self.icdar_gen = ICDAR_Gen()
        self.yvt_gen = YVT_Gen()
        self.videos_left = self.n_videos = self.icdar_gen.n_videos + self.yvt_gen.n_videos
            
        self.data_queue = []
        self.load_thread_condition = Condition()
        self.load_thread = Thread(target=self.__load_and_process_data)
        self.load_thread.start()
       
        
    def __load_and_process_data(self):
        clip_len = 8
        while self.videos_left > 0:
            if len(self.data_queue) >= 10:
                with self.load_thread_condition:
                    self.load_thread_condition.wait()
                    
            video, bbox = self.get_next_video()[1:]
            n_frames, skip_frames = video.shape[0], config.frame_skip
            
            # Skip frames after a random start
            start_loc = np.random.randint(0, skip_frames)
            # skip_vid_frames, skip_bbox_frames = [], []
            # for f in range(start_loc, n_frames, skip_frames):
            #     skip_vid_frames.append(video[f:f+1])
            #     skip_bbox_frames.append(bbox[f:f+1])
            # skip_vid = np.concatenate(skip_vid_frames, axis=0)
            # skip_bbox = np.concatenate(skip_bbox_frames, axis=0)
            skip_vid = video[start_loc::skip_frames]
            skip_bbox = bbox[start_loc::skip_frames]
            
            # Process the video into 8 frame clips
            n_frames = skip_vid.shape[0]
            for clip_start in range(0, n_frames, clip_len):
                clip = skip_vid[clip_start:clip_start+clip_len]
                bbox = skip_bbox[clip_start:clip_start+clip_len]
                
                if clip.shape[0] != clip_len:
                    print('1. clip.shape, bbox.shape', clip.shape, bbox.shape)
                    remaining_frames = clip_len - clip.shape[0]
                    clip = np.append(clip, np.zeros([remaining_frames] + list(clip.shape[1:])), axis=0)
                    bbox = np.append(bbox, np.zeros([remaining_frames] + list(bbox.shape[1:])), axis=0)
                    print('2. clip.shape, bbox.shape', clip.shape, bbox.shape)
                    
                if np.any(np.sum(bbox, axis=(1, 2, 3)) > 0):
                    self.data_queue.append((clip, bbox))
        print('[ExternalTrainDataLoader] Data Loading complete...')
        
        
    def get_batch(self, batch_size=5):
        while len(self.data_queue) == 0:
            print('[ExternalTrainDataLoader] Waiting on data')
            time.sleep(5)
            
        if self.load_thread.is_alive():
                with self.load_thread_condition:
                    self.load_thread_condition.notify_all()
        
        batch_size = min(batch_size, len(self.data_queue))
        batch_x, batch_bbox, batch_y = [], [], []
        for _ in range(batch_size):
            vid, bbox = self.data_queue.pop(0)
            batch_x.append(vid)
            bbox = np.tile(np.expand_dims(bbox, axis=1), [1, 4, 1, 1, 1])
            batch_bbox.append(bbox)
            batch_y.append(-1)
                    
        return batch_x, batch_bbox, batch_y
            
    
    def get_next_video(self):
        self.videos_left -= 1
        choice = random.randint(1, self.videos_left)
        if choice <= self.icdar_gen.videos_left:
            return self.icdar_gen.get_next_video()
        else:
            return self.yvt_gen.get_next_video()
            
            
    def has_data(self):
        return len(self.data_queue) > 0 or self.videos_left > 0
            
                
        