#!/usr/bin/python
#-*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import torch
import numpy
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from SyncNetModel import *
from shutil import rmtree
from detectors import S3FD
from scipy import signal
import numpy as np
from facetools import get_cropped_face_img, detectFace_helper, FRAME_ANALYZE_CROP_FACE_IMAGE_EXPANSION, FRAME_ANALYZE_CROP_FACE_IMAGE_TARGET_SIZE

# ==================== Get OFFSET ====================

def calc_pdist(feat1, feat2, vshift=10):
    
    win_size = vshift*2+1

    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))

    dists = []

    for i in range(0,len(feat1)):

        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))

    return dists

# ==================== MAIN DEF ====================

class SyncNetInstance(torch.nn.Module):

    def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024):
        super(SyncNetInstance, self).__init__();

        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers)
        self.DET = S3FD(device='cpu')
        self.cs = 0.4
        self.facedet_scale = 0.25
        self.blankThreshold = -1

    def computeDist(self,feat1,feat2,opt):
        dists = calc_pdist(feat1,feat2,vshift=opt.vshift)
        mdist = torch.mean(torch.stack(dists,1),1)
        
        minval, minidx = torch.min(mdist,0)
        offset = opt.vshift-minidx
        conf   = torch.median(mdist) - minval

        fdist   = numpy.stack([dist[minidx].numpy() for dist in dists])
        # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
        fconf   = torch.median(mdist).numpy() - fdist
        fconfm  = signal.medfilt(fconf,kernel_size=9)
        
        numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        dists_npy = numpy.array([ dist.numpy() for dist in dists ])
        return offset.numpy(), conf.numpy(), dists_npy, fconfm
    def evaluate(self, opt, frames):
        self.__S__.eval();
        with torch.no_grad():
            # ========== ==========
            # Convert files
            # ========== ==========

            if not os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
                os.makedirs(os.path.join(opt.tmp_dir,opt.reference))
            
            # ========== ==========
            # Load video 
            # ========== ==========

            images = frames
            im = numpy.stack(images,axis=3)
            im = numpy.expand_dims(im,axis=0)
            im = numpy.transpose(im,(0,3,4,1,2))

            imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

            # ========== ==========
            # Load audio
            # ========== ==========
            # ========== ==========
            # Check audio and video input length
            # ========== ==========


            min_length = len(images)
            
            # ========== ==========
            # Generate video and audio feats
            # ========== ==========

            lastframe = min_length-5
            im_feat = []

            tS = time.time()
            for i in range(0,lastframe,opt.batch_size):
                
                im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
                im_in = torch.cat(im_batch,0)
                im_out  = self.__S__.forward_lip(im_in);
                im_feat.append(im_out.data.cpu())
            im_feat = torch.cat(im_feat,0)
            #copy first image to be length 5 and get feature
            blank = torch.cat([imtv[:,:,0,:,:]]*5,0).swapaxes(0,1).unsqueeze(0)
            blank = self.__S__.forward_lip(blank)
            #repeat blank to be the same length as im_feat
            cc_feat = torch.cat([blank]*len(im_feat),0)
            # ========== ==========
            # Compute offset
            # ========== ==========
                

            offset, conf, dists_npy, fconfm = self.computeDist(im_feat,cc_feat,opt)
            blankLabels = fconfm < self.blankThreshold
            beginFrame = len(blankLabels)-1
            for i in range(len(blankLabels)):
                if blankLabels[i]:
                    beginFrame = i
                    break
            endFrame = 0
            for i in range(len(blankLabels)-1,-1,-1):
                if blankLabels[i]:
                    endFrame = i
                    break
            
            return im_feat, im_feat[beginFrame:endFrame], fconfm, beginFrame, endFrame, offset, conf
    def crop(self,frames):
        dets = {'x':[], 'y':[], 's':[]}
        for frame in frames:
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxes = self.DET.detect_faces(image_np, conf_th=0.9, scales=[self.facedet_scale])

            for det in bboxes:
                dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
                dets['y'].append((det[1]+det[3])/2) # crop center x 
                dets['x'].append((det[0]+det[2])/2) # crop center y
        dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
        dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
        dets['y'] = signal.medfilt(dets['y'],kernel_size=13)
        faces = []
        for bs,mx,my,frame in zip(dets['s'],dets['x'],dets['y'],frames):   
            face = frame[int(my-bs):int(my+bs*(1+2*self.cs)),int(mx-bs*(1+self.cs)):int(mx+bs*(1+self.cs))]
            if face.shape[0] == 0 or face.shape[1] == 0:
                print("face detect error")
                continue
            face = cv2.resize(face,(224,224))
            faces.append(face)
        return faces
        # croppedFrames = []
        # for roi in frames:
        #     detected = list(detectFace_helper(roi))
        #     face_photo = get_cropped_face_img(img=roi, face_bbox=detected, expansion=FRAME_ANALYZE_CROP_FACE_IMAGE_EXPANSION,
        #                                       target_size=FRAME_ANALYZE_CROP_FACE_IMAGE_TARGET_SIZE, largest=True)
        #     face_photo = cv2.resize(face_photo,(224,224))
        #     croppedFrames.append(face_photo)
        # return croppedFrames
    def extract_feature(self, opt, images):
        self.__S__.eval()
        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        
        # ========== ==========
        # Generate video feats
        # ========== ==========

        lastframe = len(images)-4
        im_feat = []

        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lipfeat(im_in);
            im_feat.append(im_out.data.cpu())

        im_feat = torch.cat(im_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========
            

        return im_feat


    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage);

        self_state = self.__S__.state_dict();

        for name, param in loaded_state.items():

            self_state[name].copy_(param);


if __name__ == "__main__":
    import time, pdb, argparse, subprocess
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
    frames = []
    import cv2
    vidCroppedPath = Path(opt.videofile).stem + "_cropped.mp4"
    vidCroppedPath = Path(opt.tmp_dir,opt.reference,vidCroppedPath)
    croppedFrames = []
    if not os.path.exists(vidCroppedPath):
        vidCap = cv2.VideoCapture(opt.videofile)
        while vidCap.isOpened():
            ret, frame = vidCap.read()
            if not ret:
                break
            frames.append(frame)
        vidCap.release()
        croppedFrames = s.crop(frames)
        vidWriter = cv2.VideoWriter(str(vidCroppedPath), cv2.VideoWriter_fourcc(*'mp4v'), 25, (croppedFrames[0].shape[1],croppedFrames[0].shape[0]))
        for frame in croppedFrames:
            vidWriter.write(frame)
        vidWriter.release()
    else:
        vidCap = cv2.VideoCapture(str(vidCroppedPath))
        while vidCap.isOpened():
            ret, frame = vidCap.read()
            if not ret:
                break
            croppedFrames.append(frame)
        vidCap.release()
    frames = croppedFrames
    # offset, conf, dists_npy, fconfm =  s.evaluate(opt, frames)
    im_feat, im_feat_trimmed, fconfm, startFrame, endFrame, offset, conf = s.evaluate(opt, frames)
    vidWriter = cv2.VideoWriter(os.path.join(opt.tmp_dir,opt.reference,f'{Path(opt.videofile).stem}_sync.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (frames[0].shape[1],frames[0].shape[0]))
    vidWriterTrimmed = cv2.VideoWriter(os.path.join(opt.tmp_dir,opt.reference,f'{Path(opt.videofile).stem}_sync_trimmed.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (frames[0].shape[1],frames[0].shape[0]))
    threshold = -1
    lenDiff = len(frames) - len(fconfm)
    #remove lenDiff frames from the beginning
    frames = frames[lenDiff:]
    for conf,image in zip(fconfm,frames):
        cv2.putText(image, '%.2f'%conf, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        vidWriter.write(image)
    vidWriter.release()
    trimmedFrames = frames[startFrame:endFrame]    
    if len(trimmedFrames) > 5:
        for image in trimmedFrames:
            vidWriterTrimmed.write(image)
        vidWriterTrimmed.release()
    else:
        print("blank video")