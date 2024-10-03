#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess

from SyncNetInstance import *

# ==================== LOAD PARAMS ====================


parser = argparse.ArgumentParser(description = "SyncNet");

parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--videofile', type=str, default="data/example.avi", help='');
parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='');
parser.add_argument('--reference', type=str, default="demo", help='');

opt = parser.parse_args();


# ==================== RUN EVALUATION ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
print("Model %s loaded."%opt.initial_model);

offset, conf, dists_npy, fconfm =  s.evaluate(opt, videofile=opt.videofile)
frames = []
import cv2
vidCap = cv2.VideoCapture(os.path.join(opt.tmp_dir,opt.reference,'video.mp4'))
while vidCap.isOpened():
    ret, frame = vidCap.read()
    if not ret:
        break
    frames.append(frame)
vidCap.release()
vidWriter = cv2.VideoWriter(os.path.join(opt.tmp_dir,opt.reference,'video_sync.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (frames[0].shape[1],frames[0].shape[0]))
vidWriterTrimmed = cv2.VideoWriter(os.path.join(opt.tmp_dir,opt.reference,'video_sync_trimmed.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (frames[0].shape[1],frames[0].shape[0]))
threshold = 0
print(os.path.join(opt.tmp_dir,opt.reference,'video_sync.mp4'))
for i in range(4,len(frames)-10):
    conf = fconfm[i-4]
    image = frames[i]
    cv2.putText(image, '%.2f'%conf, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if conf>threshold:
        vidWriterTrimmed.write(image)
    vidWriter.write(image)
vidWriter.release()
vidWriterTrimmed.release()
