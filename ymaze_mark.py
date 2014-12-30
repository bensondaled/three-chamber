import sys
import numpy as np
import os
pjoin = os.path.join
import json
import re
import itertools as it
import random as rand
import cv2
from matplotlib import path as mpl_path
from ymaze_track import FileHandler,ginput,dist,contour_center
from scipy.io import savemat

C,X,Y,Z,YEND,ZEND = 0,1,2,3,4,5 #this order is crucial!

class Marker(object):
    def __init__(self, mouse, n=1, data_dir='.'):
        self.mouse = mouse
        self.n = n
        self.data_dir = data_dir
        
        self.fh = FileHandler(self.data_dir, self.mouse, self.n)

        self.load_time()
        self.load_metadata()
        self.load_background()
        self.height, self.width = self.background.shape
        self.load_pts()
        self.make_rooms()
        self.load_tracking()

    def verify_tracking(self):
        good = [[X,C],[Y,C],[Z,C],[C,X],[C,Y],[C,Z],[Y,YEND],[YEND,Y],[Z,ZEND],[ZEND,Z]]
        for tr in self.transitions:
            if [tr['from'],tr['to']] not in good:
                raise Exception('Tracking does not make sense.')

    def end(self):
        self.results = dict(n=self.n, score=self.score, transitions=self.transitions, room_key=self.room_key, time_to_correct=self.time_to_correct, distance=self.distance, start_time=self.start_time)
        np.savez(self.fh.make_path('behaviour.npz'), **self.results)
        #savemat(self.fh.make_path('behaviour.mat',mode=0), self.results)
    def man_update(self, d):
        for k,v in d.items():
            setattr(self,k,v)
    def make_rooms(self):
        self.room_key = zip([0,1,2,3,4,5],['C=center area','X=bottom arm','Y=left arm','Z=right arm','YEND=left cup','ZEND=right cup'])
        self.path_x = mpl_path.Path(self.pts[np.array([self.xmli,self.xoli,self.xori,self.xmri])])
        self.path_y = mpl_path.Path(self.pts[np.array([self.ymli,self.yoli,self.yori,self.ymri])])
        self.path_z = mpl_path.Path(self.pts[np.array([self.zmli,self.zoli,self.zori,self.zmri])])
        self.border_mask = np.zeros((self.height,self.width))
        pth = mpl_path.Path(self.pts[np.array([self.yoli,self.yori,self.ymri,self.ycri,self.zmli,self.zoli,self.zori,self.zmri,self.zcri,self.xmli,self.xoli,self.xori,self.xmri,self.xcri,self.ymli])])
        for iy in xrange(self.border_mask.shape[0]):
            for ix in xrange(self.border_mask.shape[1]):
                self.border_mask[iy,ix] = pth.contains_point([ix,iy])

        
        zol,zml = np.array([self.pts[self.zoli],self.pts[self.zmli]]).astype(np.int32)
        dd = np.sqrt(np.sum((zol-zml)**2))
        cup_dd = 0.65*dd
        #z cup:
        zol,zml = np.array([self.pts[self.zoli],self.pts[self.zmli]]).astype(np.int32)
        d_zl = zol-zml
        theta_zl = np.arctan2(*d_zl)
        l2 = zml + cup_dd * np.array([np.sin(theta_zl),np.cos(theta_zl)])
        zor,zmr = np.array([self.pts[self.zori],self.pts[self.zmri]]).astype(np.int32)
        d_zr = zor-zmr
        theta_zr = np.arctan2(*d_zr)
        r2 = zmr + cup_dd * np.array([np.sin(theta_zr),np.cos(theta_zr)])
        self.path_endz = mpl_path.Path([zol, zor, r2, l2])
        #y cup:
        yol,yml = np.array([self.pts[self.yoli],self.pts[self.ymli]]).astype(np.int32)
        d_yl = yol-yml
        theta_yl = np.arctan2(*d_yl)
        l2 = yml + cup_dd * np.array([np.sin(theta_yl),np.cos(theta_yl)])
        yor,ymr = np.array([self.pts[self.yori],self.pts[self.ymri]]).astype(np.int32)
        d_yr = yor-ymr
        theta_yr = np.arctan2(*d_yr)
        r2 = ymr + cup_dd * np.array([np.sin(theta_yr),np.cos(theta_yr)])
        self.path_endy = mpl_path.Path([yol, yor, r2, l2])
    def load_pts(self):
        try:
            self.man_update(np.load(self.fh.make_path('pts.npz', mode=self.fh.BL)))
        except:
            raise Exception('Points file not found. Did you run tracking yet?')
    def load_time(self):
        with open(self.fh.get_path(self.fh.TRIAL,self.fh.TIME),'r') as f:
            self.time = np.array(json.loads(f.read()))
        self.Ts = np.mean(self.time[1:]-self.time[:-1])
        self.fs = 1/self.Ts
    def load_tracking(self):
        try:
            self.tracking = np.load(self.fh.make_path('tracking.npz'))
        except:
            raise Exception('It appears tracking has not yet been run!')
    def load_metadata(self):
        with open(self.fh.get_path(self.fh.TRIAL,self.fh.DATA),'r') as f:
            self.metadata = json.loads(f.read())
        try:
            self.side = self.metadata['side']
        except:
            self.side = None
            while self.side not in ['l','r']:
                self.side = raw_input('Correct side was not indicated in this dataset. Please enter correct side (l/r):').lower()
            self.metadata['side'] = self.side
            try:
                with open(self.fh.get_path(self.fh.TRIAL,self.fh.DATA),'w') as f:
                    towrite = json.dumps(self.metadata)
                    f.write("%s"%towrite)
            except: #backup in an attempt to never lose original data
                print self.metadata
                with open('crash_log.txt','a') as f:
                    f.write("%s"%str(self.metadata))
        self.correct = [Y,Z][['l','r'].index(self.side)]
        self.incorrect = [Z,Y][['l','r'].index(self.side)]
    def load_background(self):
        try:
            bg = np.load(self.fh.make_path('background.npz',mode=self.fh.BL))
            background = bg['computations']
            background_image = bg['image']
        except:
            raise Exception('Background file not found. Did you run tracking yet?')
        self.background, self.background_image = background, background_image
    def get_chamber(self, pos):
        chamber = C
        if self.path_endy.contains_point(pos):
            chamber = YEND
        elif self.path_endz.contains_point(pos):
            chamber = ZEND
        elif self.path_x.contains_point(pos):
            chamber = X
        elif self.path_y.contains_point(pos):
            chamber = Y
        elif self.path_z.contains_point(pos):
            chamber = Z
        return chamber
    def total_distance(self, arr):
        return np.sum(np.array([dist(*i) for i in zip(arr[1:],arr[:-1])]))
    def run(self):
        #correct for proper start time:
        for idx,p in enumerate(self.tracking['pct_xadj'][:20]):
            if p<0.2:
                break
        start_idx = idx
        self.start_time = self.tracking['time'][start_idx]
        time = self.tracking['time'][start_idx:]
        pos = self.tracking['pos'][start_idx:]
        
        #run
        self.distance = self.total_distance(pos)
        self.chamber = np.array(map(self.get_chamber, pos))
        durations = time[1:] - time[0]
        moved = self.chamber[1:]-self.chamber[:-1]
        self.transitions = np.array(zip(durations[moved != 0], self.chamber[moved!=0], self.chamber[1:][moved != 0]), dtype=[('time',float),('from',int),('to',int)]) #time, chamber exited,  chamber entered
        self.verify_tracking()

        self.score = ''
        for t in self.transitions:
            if self.score == 'null':
                if t['to'] == self.correct+2:
                    self.score = 'correct'
                break
            if t['to'] == self.correct:
                self.score = 'null'
                continue
            if t['to'] == self.incorrect:
                self.score = 'incorrect'
                break
        if self.score == 'correct':
            self.time_to_correct = t['time']
        else:
            self.time_to_correct = -1

        self.end()

if __name__ == '__main__':
    data_dir = '/Users/Benson/Desktop/'
    mouse = 'DREADD_GR3_M1_acq1'
    mouse = 'DREADD_GR3_M1_revD1_1'

    m = Marker(mouse=mouse, n=1, data_dir=data_dir)
    m.run()
