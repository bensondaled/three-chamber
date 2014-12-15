import numpy as np 
import time
from cameras import Camera
import os
import json
import cv2
cv = cv2.cv


class Monitor(object):
    def __init__(self, camera=None, show=True, save_on=True, run_name='', dirr='', duration=99999999999.,extra={}):
        if type(camera) != Camera:
            raise Exception('Camera object is not a valid camera!')
        self.cam = camera
        self.show = show
        self.save_on = save_on
        cv2.namedWindow('Camera', cv.CV_WINDOW_NORMAL)
        self.duration = duration
        if self.save_on:
            self.time = []
            self.run_time = time.strftime("%Y%m%d_%H%M%S")
            self.run_name = run_name
            self.dirr = dirr
            if self.run_name == '':
                self.run_name = self.run_time
            if self.dirr == '':
                self.dirr = self.run_name
                
            if not os.path.exists(self.dirr):
                os.mkdir(self.dirr)
            
            avi_file = os.path.join(self.dirr, self.run_name+'-cam.avi')
            self.writer = cv2.VideoWriter(avi_file,\
            cv.CV_FOURCC('M','J','P','G'),\
            self.cam.frame_rate,\
            frameSize=self.cam.resolution,\
            isColor=self.cam.color_mode)
            
            dic = {}
            dic['dirr'] = self.dirr
            dic['run_name'] = self.run_name
            dic['run_time'] = self.run_time
            dic['cameras'] = self.cam.metadata()
            dic.update(extra)
            
            meta_name = os.path.join(self.dirr, self.run_name+'-metadata.json')
            with open(meta_name, 'w') as f:
                f.write("%s"%json.dumps(dic))
        
    def end(self):
        cv2.destroyAllWindows()
        if self.save_on:
            self.writer.release()
                        
            time_name = os.path.join(self.dirr, self.run_name+'-timestamps.json')
            with open(time_name, 'w') as f:
                f.write("%s"%json.dumps(self.time))
    def next_frame(self, t0):
        frame, timestamp = self.cam.read()
        if self.show:
            showframe = frame.copy()
            cv2.putText(showframe, "%0.3f"%(timestamp-t0), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
            cv2.imshow('Camera', showframe)
            c = cv2.waitKey(1)
        if self.save_on:
            self.time.append(timestamp)
            self.writer.write(frame)
        return c
    
    def go(self):
        t = time.time()
        c = None
        while time.time()-t < self.duration and c!=ord('q'):
            c = self.next_frame(t)
        c = self.next_frame(t)
        self.end()

if __name__ == '__main__':
    m = Monitor(cameras=Camera(), show=True, save_on=False)
    m.go()
