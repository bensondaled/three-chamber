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
import warnings

def dist_pl(p, p1, p2):
    #dist from point to line
    p,p1,p2 = map(np.array, [p,p1,p2])
    x1,y1 = p1
    x2,y2 = p2
    return np.abs((y2-y1)*p[0] - (x2-x1)*p[1] + x2*y1 - y2*x1)/np.sqrt(np.sum((p2-p1)**2))

C,X,Y0,Z0,Y,Z,YEND,ZEND = 0,1,2,3,4,5,6,7 #this order is crucial!

class Marker(object):
    def __init__(self, mouse, n=1, data_dir='.', consecutive_threshold=0.200, cup_dd_factor=0.54, mark_mode='normal'):
        self.mouse = mouse
        self.n = n
        self.data_dir = data_dir
        self.consecutive_threshold = consecutive_threshold #seconds
        self.cup_dd_factor = cup_dd_factor
        self.mark_mode = mark_mode #determines only the result of correct/incorrect/etc. all other results *always* use normal mode
        
        self.fh = FileHandler(self.data_dir, self.mouse, self.n)

        self.load_time()
        self.load_metadata()
        self.load_background()
        self.height, self.width = self.background.shape
        self.load_pts()
        self.make_rooms()
        self.load_tracking()

    def verify_tracking(self):
        good = [[X,C],[C,X],[C,Y0],[Y0,C],[Y0,Y],[Y,Y0],[Y,YEND],[YEND,Y],[C,Z0],[Z0,C],[Z0,Z],[Z,Z0],[Z,ZEND],[ZEND,Z]]
        for tr in self.transitions:
            if [tr['from'],tr['to']] not in good:
                return (False, 'Tracking does not make sense.')
        return (True,'')

    def end(self):
        self.results = dict(n=self.n, score=self.score, transitions=self.transitions, room_key=self.room_key, time_to_correct=self.time_to_correct, distance=self.distance, start_time=self.start_time, start_idx=self.start_idx, mark_mode=self.mark_mode, chamber=self.chamber)
        np.savez(self.fh.make_path('behaviour.npz',mode=1), **self.results)
        savemat(self.fh.make_path('behaviour.mat',mode=1), self.results)
    def man_update(self, d):
        for k,v in d.items():
            setattr(self,k,v)
    def make_rooms(self):
        self.room_key = zip([0,1,2,3,4,5,6,7],['C=center area','X=bottom arm','Y0=left arm central', 'Z0=right arm central','Y=left arm','Z=right arm','YEND=left cup','ZEND=right cup'])
        self.path_x = mpl_path.Path(self.pts[np.array([self.xmli,self.xoli,self.xori,self.xmri])])
        self.path_y = mpl_path.Path(self.pts[np.array([self.ymli,self.yoli,self.yori,self.ymri])])
        self.path_z = mpl_path.Path(self.pts[np.array([self.zmli,self.zoli,self.zori,self.zmri])])
        self.path_x_full = mpl_path.Path(self.pts[np.array([self.xcli,self.xoli,self.xori,self.xcri])])
        self.path_y_full = mpl_path.Path(self.pts[np.array([self.ycli,self.yoli,self.yori,self.ycri])])
        self.path_z_full = mpl_path.Path(self.pts[np.array([self.zcli,self.zoli,self.zori,self.zcri])])
        self.border_mask = np.zeros((self.height,self.width))
        pth = mpl_path.Path(self.pts[np.array([self.yoli,self.yori,self.ymri,self.ycri,self.zmli,self.zoli,self.zori,self.zmri,self.zcri,self.xmli,self.xoli,self.xori,self.xmri,self.xcri,self.ymli])])
        for iy in xrange(self.border_mask.shape[0]):
            for ix in xrange(self.border_mask.shape[1]):
                self.border_mask[iy,ix] = pth.contains_point([ix,iy])

        
        zol,zml = np.array([self.pts[self.zoli],self.pts[self.zmli]]).astype(np.int32)
        dd = np.sqrt(np.sum((zol-zml)**2))
        cup_dd = self.cup_dd_factor*dd #bigger multiplier means smaller end zone
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
                print 'Warning for mouse %s #%i :'%(self.mouse, self.n)
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
        self.incorrect = [Y,Z][['r','l'].index(self.side)]
        self.destination = self.correct+2
        if self.mark_mode == 'normal':
            self.correct_pnr = self.correct #point of no return
            self.incorrect_pnr = self.incorrect #point of no return
        elif self.mark_mode == 'extend':
            self.correct_pnr = self.correct-2
            self.incorrect_pnr = self.incorrect-2
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
        elif self.path_y_full.contains_point(pos):
            chamber = Y0
        elif self.path_z_full.contains_point(pos):
            chamber = Z0
        return chamber
    def total_distance(self, arr):
        return np.sum(np.array([dist(*i) for i in zip(arr[1:],arr[:-1])]))
    def correct_for_consecutive(self, tr):
        if len(tr)==0:
            return tr
        skipnext = False
        new_tr = [tr[0]]
        for t,last,nex in zip(tr[1:-1],tr[:-2],tr[2:]):
            if skipnext:
                skipnext = False
                continue
            if t['to']==last['from'] and nex['to']==t['from'] and nex['time']-last['time']<=self.consecutive_threshold:
                skipnext = True
            else:
                new_tr.append(t)
        new_tr.append(tr[-1])
        return np.array(new_tr)
    def run(self, resample=1, thresh_p_hand=0.001,thresh_wall_dist=6,start_range=260,thresh_wall_dist_x=0.33, start_time='auto'):
        thresh_wall_dist_x = thresh_wall_dist_x*dist(self.pts[self.xoli],self.pts[self.xmli])
        #correct for proper start time:
        if start_time == 'auto':
            if np.any(self.tracking['pct_xadj'][:start_range]):
                started = False
                for idx,c,p in zip(range(260),self.tracking['contour'][:start_range],self.tracking['pct_xadj'][:start_range]):
                    mindist_r = min([dist_pl(np.squeeze(ppp),self.pts[self.xori],self.pts[self.xmri]) for ppp in c])
                    mindist_l = min([dist_pl(np.squeeze(ppp),self.pts[self.xoli],self.pts[self.xmli]) for ppp in c])
                    mindist_b = min([dist_pl(np.squeeze(ppp),self.pts[self.xoli],self.pts[self.xori]) for ppp in c])
                    onwall = mindist_b<=thresh_wall_dist_x and (mindist_r<=thresh_wall_dist or mindist_l<=thresh_wall_dist)
                    if not started and (p>thresh_p_hand or onwall):
                        started = True
                    if started and p<thresh_p_hand and (not onwall):
                        break
            else:
                idx = -1
            start_idx = idx+1
            self.start_idx = start_idx
        elif start_time == None:
            start_idx = 0
            self.start_idx = 0
        elif type(start_time) in [int,float]:
            dsts = np.abs(self.tracking['time']-start_time)
            start_idx = np.argmin(dsts)
            self.start_idx = start_idx
        self.start_time = self.tracking['time'][start_idx]
        time = self.tracking['time'][start_idx:]
        pos = self.tracking['pos'][start_idx:]

        #resample
        time = time[::resample]
        pos = pos[::resample]
        
        #run
        self.distance = self.total_distance(pos)
        self.chamber = np.array(map(self.get_chamber, pos))
        durations = time[1:] - time[0]
        moved = self.chamber[1:]-self.chamber[:-1]
        self.transitions = np.array(zip(durations[moved != 0], self.chamber[:-1][moved!=0], self.chamber[1:][moved != 0]), dtype=[('time',float),('from',int),('to',int)]) #time, chamber exited,  chamber entered
        self.transitions = self.correct_for_consecutive(self.transitions)
        verif = self.verify_tracking()

        self.score = 'none'
        for t in self.transitions:
            if self.score == 'null':
                if t['to'] not in [self.correct, self.correct_pnr, self.destination, C]:
                    break
                elif t['to'] == self.destination:
                    self.score = 'correct'
                    tcor = t['time']
                    break
            elif t['to'] == self.correct_pnr:
                self.score = 'null'
                continue
            elif t['to'] == self.incorrect_pnr:
                self.score = 'incorrect'
                break

        if self.score == 'correct':
            self.time_to_correct = tcor
        else:
            self.time_to_correct = -1

        self.end()
        return verif

if __name__ == '__main__':
    data_dir = '/Volumes/wang/abadura/Y-Maze_analyzed/DREADDs'
    #data_dir = '/Users/Benson/Desktop/'
    mouse = 'DREADD_GR5_M3_hab'

    m = Marker(mouse=mouse, n=2, data_dir=data_dir)
    m.run(start_time=1.0)
