import json
import cv2
import numpy as np
import os
import csv
import tkFileDialog as tk
from ymaze_track import FileHandler

class Playback(object):
    def __init__(self, ymaze_path=None, ymaze_n=1, movfile='', timefile='', data_dir='.'):
        self.ymaze_path = ymaze_path
        self.ymaze_n = ymaze_n
        self.data_dir = data_dir
        self.movfile = os.path.join(self.data_dir,movfile)
        self.timefile = os.path.join(self.data_dir,timefile)

        if self.ymaze_path != None:
            self.fh = FileHandler(self.data_dir, self.ymaze_path, self.ymaze_n)
            self.movfile = self.fh.get_path(self.fh.TRIAL, self.fh.MOV, self.ymaze_n)
            self.timefile = self.fh.get_path(self.fh.TRIAL, self.fh.TIME, self.ymaze_n)

        self.t = np.squeeze(json.loads(open(self.timefile).read()))
        self.Ts = int(np.rint(np.mean(self.t[1:]-self.t[:-1])*1000))
        self.mov = cv2.VideoCapture(self.movfile)
        _=self.mov.read()
        self.t = self.t[1:]
    def play(self):
        idx = 0
        cv2.namedWindow('Playback')
        cv2.createTrackbar('Ts', 'Playback', self.Ts, 200, lambda x: x)
        paused = True

        valid,frame = self.mov.read()
        while valid:
            delay = cv2.getTrackbarPos('Ts', 'Playback')
            if delay == 0:
                delay = 1
            if not paused:
                tstr = "%0.3f"%(self.t[idx]-self.t[0])
                frame = frame.astype(np.uint8)
                cv2.putText(frame, tstr, (0,frame.shape[0]-3), cv2.FONT_HERSHEY_SIMPLEX, 1., (255,255,255))
                cv2.imshow('Playback', frame)
                idx += 1
            k = cv2.waitKey(delay)
            if k == ord('p'):
                paused = not paused
            elif k == ord('q'):
                break
            valid,frame = self.mov.read()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    f = tk.askopenfilename('Select movie file.')
    t = tk.askopenfilename('Select time file.')
    pb = Playback(movfile=f, timefile=t)
    pb.play()


