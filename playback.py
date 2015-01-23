import json
import cv2
import numpy as np
import os
import csv
import tkFileDialog as tk
from ymaze_track import FileHandler

class Playback(object):
    def __init__(self, ymaze_path=None, ymaze_n=1, movfile=None, timefile=None, data_dir='.'):
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
        Ts = int(np.rint(np.mean(self.t[1:]-t[:-1])*1000))
        self.mov = cv2.VideoCapture(self.movfile)
        _=self.mov.read()
        self.t = self.t[1:]
    def play(self):
        idx = 0
        cv2.namedWindow('Movie')
        cv2.createTrackbar('Ts', 'Playback', Ts, 200, lambda x: x)
        paused = True

        valid,frame = mov.read()
        while valid:
            if not paused:
                tstr = "%0.3f"%(t[idx]-self.t[0])
                cv2.putText(frame, tstr, (0,frame.shape[0]-3), cv2.FONT_HERSHEY_SIMPLEX, 1., (255,255,255))
                cv2.imshow('Playback', frame)
                delay = cv2.getTrackbarPos('Ts', 'Playback')
                if delay == 0:
                    delay = 1
                tidx += 1
            k = cv2.waitKey(delay)
            if k == ord('p'):
                paused = not paused
            elif k == ord('q'):
                break
            valid,frame = mov.read()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    f = tk.askopenfilename('Select movie file.')
    t = tk.askopenfilename('Select time file.')
    pb = Playback(movfile=f, timefile=t)
    pb.play()


