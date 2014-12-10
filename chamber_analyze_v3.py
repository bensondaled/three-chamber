import numpy as np
import pylab as pl
import cv2
import sys
import os
pjoin = os.path.join

trial_path = pjoin(data_dir,mouse,'%s-cam.avi'%mouse)
bg_path = pjoin(data_dir,mouse,'%s_BL-cam.avi'%mouse)

vc = cv2.VideoCapture(trial_path)
count = 0
valid,frame = vc.read()
mean_frame = frame.astype(np.uint32)
while valid:
    mean_frame += frame
    count += 1
    valid,frame = vc.read()
mean_frame /= count
mean_frame = mean_frame.astype(np.uint8)

hsv = cv2.cvtColor(mean_frame, cv2.cv.CV_BGR2HSV)
spots = np.argwhere(abs(hsv[...,0] - 32.) < 1.1)
mean_frame_filt = mean_frame.copy()
mean_frame_filt[spots[:,0],spots[:,1],:] = 0
cv2.imshow('a',mean_frame_filt)

if __name__ == '__main__':
    data_dir = '/Users/Benson/Desktop/mouse/'
    mouse = 'black-1'
    tr = Tracker(mouse=mouse, data_dir=data_dir)
