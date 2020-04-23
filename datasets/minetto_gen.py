import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw 

base_dir = '/home/shivam/Downloads/minetto/www.liv.ic.unicamp.br/~minetto/datasets/text/VIDEOS/'
rect_prop_list = ['x', 'y', 'w', 'h', 'text', 'vfr']
in_h, in_w = 480, 640
out_h, out_w = 256, 480
n_videos = 5

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
    '''
    Returns a dict which is something like:
    {image_num:{rectangle_id: rectangle_properties, ...}, ...}
    '''
    tree = ET.parse(file)
    root = tree.getroot()
    ann = {}
    for image in root.findall('./image'):
        image_num = int(image.find('imageName').text)
        rectangle = {}
        for rectangle_tag in image.findall('./taggedRectangles/taggedRectangle'):
            id = int(rectangle_tag.attrib['id'])
            properties = {}
            for prop_name in rect_prop_list:
                if prop_name == 'text':
                    properties[prop_name] = rectangle_tag.attrib[prop_name]
                else:
                    properties[prop_name] = int(float(rectangle_tag.attrib[prop_name]))
            rectangle[id] = properties
        ann[image_num] = rectangle
    return ann

def create_mask(pts):
    mask = np.zeros((in_h, in_w), dtype=np.uint8)
    mask = Image.fromarray(mask, 'L')
    draw = ImageDraw.Draw(mask)
    for pt in pts:
        draw.rectangle(pt.tolist(), fill=1)
    del draw
    mask = np.asarray(mask).copy()
    return mask
    

def get_pts_for_rectangles(rectangles):
    num_rectangles = len(rectangles)
    rect_pts = np.zeros((num_rectangles, 4,), dtype=np.int32)
    for rect_id, rect_prop in rectangles.items():
        rect_pts[rect_id] = rect_prop['x'], rect_prop['y'], rect_prop['x']+rect_prop['w'], rect_prop['y']+rect_prop['h']
    return rect_pts
        

def minetto_gen():
    for video_dir in filter(lambda x: os.path.isdir(base_dir+x),os.listdir(base_dir)):
        ann_file = base_dir+video_dir+'/groundtruth.xml'
        ann = parse_ann(ann_file)
        
        image_dir = base_dir+video_dir+'/PNG/'
        n_frames = len(os.listdir(image_dir))
        video = np.zeros((n_frames, out_h, out_w, 3), dtype=np.uint8)
        mask = np.zeros((n_frames, out_h, out_w, 1), dtype=np.uint8)
        
        for idx in range(n_frames):
            frame_loc = image_dir + '%06d.png' % idx
            frame = cv2.cvtColor(cv2.imread(frame_loc), cv2.COLOR_BGR2RGB)
            video[idx] = resize_and_pad(frame)
            
            if idx in ann:
                rectangles = ann[idx]
                pts = get_pts_for_rectangles(rectangles)
                frame_mask = create_mask(pts)
                mask_resized = resize_and_pad(frame_mask)
                mask[idx] = np.expand_dims(np.array(mask_resized), axis=-1)
        
        yield video_dir, video/255., mask

    