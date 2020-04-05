import os
import time
import numpy as np
import random
from threading import Thread
import h5py
import sys
from scipy.misc import imread
from PIL import Image, ImageDraw
import cv2
import config
import traceback

#dataset_dir = '../../data/SyntheticVideos/'
dataset_dir = '../SynthVideo/out/'

#bad_files = ['9410_tr_t_b_','9535_tr_l_r_']
bad_files = []

def get_det_annotations(split='train'):
    """
    Loads in all annotations for training and testing splits. This is assuming the data has been loaded in with resized frames (half in each dimension).
    :return: returns two lists (one for training and one for testing) with the file names and annotations. The lists
    contain tuples with the following content (file name, annotations), where annotations is a list of tuples with the
    form (start frame, end frame, label, bounding boxes).
    """
    polygon_ann = []
    with h5py.File(dataset_dir + 'Annotations/synthvid_ann.hdf5', 'r') as hf:
        for label in hf.keys():
            label_grp = hf.get(label)
            for file in label_grp.keys():
                if file not in bad_files:
                    file_grp = label_grp.get(file)
                    k = label + '/' + file
                    v = {'label': int(label),
                        #'char_ann': file_grp.get('char_ann')[()],
                        #'word_ann': file_grp.get('word_ann')[()],
                        #'line_ann': file_grp.get('line_ann')[()],
                        'para_ann': np.rint(np.array(file_grp.get('para_ann')[()]) / 2).astype(np.int32)
                        }
                    #print(label)
                    polygon_ann.append((k, v))
    random.seed(7)
    random.shuffle(polygon_ann)
    num_samples = len(polygon_ann)
    num_train_samples = int(0.8 * num_samples)
    num_test_samples = num_samples - num_train_samples
    
    if split == 'train':
        train_split = polygon_ann[:num_train_samples]
        random.seed()
        random.shuffle(train_split)
        print("Num train samples:", len(train_split))
        return train_split
    elif split == 'test':
        test_split = polygon_ann[-num_test_samples:]
        print("Num test samples:", len(test_split))
        return test_split
    


def create_mask(shape, pts):
    im = np.zeros(shape, dtype=np.uint8)
    im = Image.fromarray(im, 'L')
    draw = ImageDraw.Draw(im)
    draw.polygon(pts.tolist(), fill=1)
    del draw
    # print(pts.tolist())
    #input()
    im = np.asarray(im).copy()
    #im = im // 255
    #print('np.min(im)', np.min(im))
    #print('np.max(im)', np.max(im))
    #print(im.dtype)
    #cv2.imwrite('temp2.jpg', im)
    #input()
    return np.reshape(im, im.shape + (1,))


def get_video_det(video_dir, annotations, skip_frames=1, start_rand=True):
    """
    Loads in a video.

    :param video_dir: the directory of the video
    :param annotations: the annotations for that video
    :param skip_frames: number of frames which can be skipped
    :param start_rand: if True, then the first frame can be random (only useful if using skip_frames > 1). If False, then the first frame of the video will be the first frame of the video.
    :return: returns the video, bounding box annotations, and label
    """

    n_frames = len(os.listdir(video_dir))
    frame_start = 0

    im0 = imread(video_dir + ('frame_%d.jpg' % frame_start))
    h, w, ch = im0.shape
    video = np.zeros((n_frames, config.vid_h, config.vid_w, ch), dtype=np.uint8)
    bbox = np.zeros((n_frames, config.vid_h, config.vid_w, 1), dtype=np.uint8)
    label = annotations['label']
    
    # video[0] = im0
    # annotations['para_ann'][count] has type [[0 0...]]
    for idx in range(n_frames):
        frame = imread(video_dir + ('frame_%d.jpg' % idx))
        
        if (h, w, ch) != frame.shape:
            print('BAD FRAMES FOUND')
            print('Video:', video_dir)
            print('Frame:', idx)
            print('*' * 20)
            frame = cv2.resize(frame, (w, h))
        # print(annotations['para_ann'][idx,0])
        mask = create_mask((config.vid_h, config.vid_w), annotations['para_ann'][idx,0])
        # input()
        '''
        mask = cv2.resize(mask, (config.vid_w, config.vid_h))
        mask = np.reshape(mask, mask.shape + (1,))
        '''
        frame = cv2.resize(frame, (config.vid_w, config.vid_h))
        video[idx] = frame
        bbox[idx] = mask  
        
        
    if skip_frames == 1:
        return video, bbox, label

    skip_vid_frames, skip_bbox_frames = [], []
    if start_rand:
        start_frame = np.random.randint(0, skip_frames)
    else:
        start_frame = 0

    for f in range(start_frame, n_frames, skip_frames):
        skip_vid_frames.append(video[f:f+1])
        skip_bbox_frames.append(bbox[f:f+1])

    skip_vid = np.concatenate(skip_vid_frames, axis=0)
    skip_bbox = np.concatenate(skip_bbox_frames, axis=0)

    return skip_vid, skip_bbox, label


def get_clip_det(video, bbox, clip_len=8, any_clip=False):
    """
    Creates a clip from a video.

    :param video: The video
    :param bbox: The bounding box annotations
    :param clip_len: The length of the clip
    :param any_clip: If True, then any clip from the video can be used. If False only clips with bounding box
    annotations in all frames can be used.
    :return: returns the video clip and the bounding box annotations clip
    """
    
    if any_clip:
        n_frames, _, _, _ = video.shape
        start_loc = np.random.randint(0, n_frames-clip_len)
    else:
        frame_anns = np.sum(bbox, axis=(1, 2, 3))
        ann_locs = np.where(frame_anns > 0)[0]
        ann_locs = [i for i in ann_locs if (i+clip_len-1) in ann_locs]

        try:
            start_loc = ann_locs[np.random.randint(0, len(ann_locs))]
        except:
            start_loc = min(np.where(frame_anns > 0)[0][0], video.shape[0]-clip_len)

    return video[start_loc:start_loc+clip_len]/255., bbox[start_loc:start_loc+clip_len]


def crop_clip_det(clip, bbox_clip, crop_size=(112, 112), shuffle=True):
    """
    Crops the clip to a given spatial dimension

    :param clip: the video clip
    :param bbox_clip: the bounding box annotations clip
    :param crop_size: the size which the clip will be cropped to
    :param shuffle: If True, a random cropping will occur. If False, a center crop will be taken.
    :return: returns the cropped clip and the cropped bounding box annotation clip
    """

    frames, h, w, _ = clip.shape
    if not shuffle:
        margin_h = h - crop_size[0]
        h_crop_start = int(margin_h/2)
        margin_w = w - crop_size[1]
        w_crop_start = int(margin_w/2)
    else:
        h_crop_start = np.random.randint(0, h - crop_size[0])
        w_crop_start = np.random.randint(0, w - crop_size[1])

    return clip[:, h_crop_start:h_crop_start+crop_size[0], w_crop_start:w_crop_start+crop_size[1], :] / 255., \
           bbox_clip[:, h_crop_start:h_crop_start+crop_size[0], w_crop_start:w_crop_start+crop_size[1], :]

# The data generator for training. Outputs clips, bounding boxes, and labels for the training split.
class SynthTrainDataGenDet(object):
    def __init__(self, sec_to_wait=5, frame_skip=2):
        self.train_files = get_det_annotations()
        self.sec_to_wait = sec_to_wait
        self.frame_skip = frame_skip
        self.frames_dir = dataset_dir + "Frames/"

        np.random.seed(None)
        # random.shuffle(self.train_files)

        self.data_queue = []

        self.load_thread = Thread(target=self.__load_and_process_data)
        self.load_thread.start()

        print('Running SynthTrainDataGenDet...')
        print('Waiting %d (s) to load data' % sec_to_wait)
        time.sleep(self.sec_to_wait)

    def __load_and_process_data(self):
        while self.train_files:
            while len(self.data_queue) >= 600:
                time.sleep(1)
            vid_name, anns = self.train_files.pop()
            while True:
                try:
                    clip, bbox_clip, label = get_video_det(self.frames_dir + vid_name + '/', 
                                                           anns, skip_frames=self.frame_skip, start_rand=True)
                    clip, bbox_clip = get_clip_det(clip, bbox_clip, any_clip=False)
                    # clip, bbox_clip = crop_clip_det(clip, bbox_clip, crop_size=(config.vid_h, config.vid_w), shuffle=True)
                    self.data_queue.append((clip, bbox_clip, label))
                    break
                except Exception as e:
                    print('Unexpected error:', sys.exc_info()[0])
                    print('Video:', vid_name)
                    print(traceback.format_exc())
                    print('*' * 20)
                    if self.train_files:
                        vid_name, anns = self.train_files.pop()
                    else:
                        break
        print('Loading data thread finished')

    def get_batch(self, batch_size=5):
        while len(self.data_queue) < batch_size and self.train_files:
            print('Waiting on data')
            time.sleep(self.sec_to_wait)

        batch_size = min(batch_size, len(self.data_queue))
        batch_x, batch_bbox, batch_y = [], [], []
        for i in range(batch_size):
            vid, bbox, label = self.data_queue.pop(0)
            batch_x.append(vid)
            batch_bbox.append(bbox)
            batch_y.append(label)

        return batch_x, batch_bbox, batch_y

    def has_data(self):
        return self.data_queue != [] or self.train_files != []

# The data generator for testing. Outputs clips, bounding boxes, and labels for the testing split.
class SynthTestDataGenDet(object):
    def __init__(self, sec_to_wait=5, frame_skip=1):
        self.test_files = get_det_annotations(split='test')
        self.n_videos = len(self.test_files)
        self.sec_to_wait = sec_to_wait
        self.skip_frame = frame_skip
        self.frames_dir = dataset_dir + 'Frames/'

        self.video_queue = []
        self.videos_left = self.n_videos
        self.data_queue = []

        self.load_thread = Thread(target=self.__load_and_process_data)
        self.load_thread.start()

        print('Running SynthTestDataGenDet...')
        print('Waiting %d (s) to load data' % sec_to_wait)
        time.sleep(self.sec_to_wait)

    def __load_and_process_data(self):
        while self.test_files:
            while len(self.data_queue) >= 50:
                time.sleep(1)
            vid_name, anns = self.test_files.pop(0)
            while True:
                try:
                    clip, bbox_clip, label = get_video_det(self.frames_dir + vid_name + '/', anns, skip_frames=self.skip_frame, start_rand=False)
                    clip, bbox_clip = get_clip_det(clip, bbox_clip, any_clip=False)
                    # clip, bbox_clip = crop_clip_det(clip, bbox_clip, crop_size=(config.vid_h, config.vid_w), shuffle=False)
                    self.data_queue.append((clip, bbox_clip, label))
                    break
                except Exception as e:
                    print('Unexpected error:', sys.exc_info()[0])
                    print(traceback.format_exc())
                    print('Video:', vid_name)
                    print('*' * 20)
                    if self.test_files:
                        vid_name, anns = self.test_files.pop(0)
                    else:
                        break
        print('Loading data thread finished')

    def get_next_video(self):
        while len(self.data_queue) == 0:
            print('Waiting on data')
            time.sleep(self.sec_to_wait)

        return self.data_queue.pop(0)

    def has_data(self):
        return self.data_queue != [] or self.test_files != []
