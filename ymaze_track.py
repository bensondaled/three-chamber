# For the setup implemented with the Y-maze. baseline now exist in same directory, etc.

#natives
import sys
import os
pjoin = os.path.join
import json
import re
import itertools as it
import random as rand

#numpy & scipy
from scipy.io import savemat
import numpy as np

#matplotlib
import pylab as pl
from matplotlib import path as mpl_path
from pylab import ion, title, close,scatter
from pylab import imshow as plimshow
from pylab import ginput as plginput
from pylab import savefig as plsavefig
from pylab import close as plclose
import matplotlib.cm as mpl_cm
pl.ion()

#tkinter
from tkFileDialog import askopenfilename, askdirectory
import ttk
import Tkinter as tk

#opencv
from cv2 import getRotationMatrix2D, warpAffine, namedWindow, VideoCapture, destroyAllWindows, cvtColor, GaussianBlur, VideoWriter, absdiff, threshold, THRESH_BINARY, Canny, findContours,  RETR_EXTERNAL, CHAIN_APPROX_TC89_L1, contourArea, circle, waitKey, resize
from cv2 import imshow as cv2imshow
from cv2.cv import CV_RGB2GRAY, CV_FOURCC, CV_GRAY2RGB
import cv2

BASELINE = 0
TRIAL = 1
DIR = 0
NAME = 1

def ginput(n):
    try:
        pts = plginput(n, timeout=-1)
        pts = np.array(pts)
    except:
        pts = np.array([])
    return pts
def contour_center(c, asint=False):
    res = np.mean(c, axis=0)
    if asint:
        res = np.round(res).astype(int)
    return res
def dist(pt1, pt2):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    return np.sqrt(np.sum((pt1-pt2)**2))

class FileHandler(object):
    MOV = 0
    TIME = 1
    DATA = 2
    TRIAL = 1
    BL = 0
    def __init__(self, data_dir, mouse, n):
        self.data_dir = data_dir
        self.mouse = mouse
        self.n = n
        try:
            self.dirr = os.path.join(data_dir,mouse)
            self.contents = os.listdir(self.dirr)
        except:
            print "File name handling failed."
    def get_path(self, blm, mode, n=None):
        #mode is movie/timestamps/metadata
        #n is trial number, 0 refers to baseline
        if n == None:
            n = self.n
        filename = self.mouse
        if blm == self.BL:
            filename += '_BL'
        elif blm == self.TRIAL:
            filename += "_%02d_"%(n)
        if mode == self.MOV:
            filename += '-cam.avi'
        elif mode == self.TIME:
            filename += '-timestamps.json'
        elif mode == self.DATA:
            filename += '-metadata.json'
        pth = pjoin(self.dirr, filename)
        return pth
    def make_path(self, st, mode=1, n=None):
        if n == None:
            n = self.n
        filename = self.mouse
        if mode == self.BL:
            filename += '_'
        elif mode == self.TRIAL:
            filename += "_%02d_"%(n)
        pth = pjoin(self.dirr, filename+st)
        return pth
    def get_n_trials(self):
        i = 1
        exists = True
        while exists:
            if not os.path.exists(self.get_path(self.TRIAL,self.MOV,n=i)):
                break
            i+=1
        return i-1
    def get_n_trials_wbehav(self):
        i = 1
        exists = True
        while exists:
            if not os.path.exists(self.make_path('behaviour.npz',n=i)):
                break
            i+=1
        return i-1

class MouseTracker(object):
    def __init__(self, mouse, n=1, data_dir='.', diff_thresh=80, resample=1, translation_max=50, smoothing_kernel=19, consecutive_skip_threshold=0.08, selection_from=[], point_mode='auto'):
        self.mouse = mouse
        self.n = n
        self.data_dir = data_dir
        
        # Parameters (you may vary)
        self.diff_thresh = diff_thresh
        self.resample = resample
        self.translation_max = translation_max
        self.kernel = smoothing_kernel

        # Parameters (you should not vary)
        self.cth1 = 0
        self.cth2 = 0
        plat = sys.platform
        if 'darwin' in plat:
            self.fourcc = CV_FOURCC('m','p','4','v') 
        elif plat[:3] == 'win':
            self.fourcc = 1
        else:
            self.fourcc = -1
        
        self.fh = FileHandler(self.data_dir, self.mouse, self.n)

        self.framei = 0
        self.load_time()
        self.consecutive_skip_threshold = (self.fs/self.resample) * consecutive_skip_threshold
        self.load_background()
        self.height, self.width = self.background.shape
        self.mov = VideoCapture(self.fh.get_path(self.fh.TRIAL, self.fh.MOV))
        self.mov.read();self.time=self.time[1:]
        #self.get_frame(self.mov,n=40) #MUST ADJUST TIME IF USING THIS
        self.load_pts(mode=point_mode)
        self.make_rooms()

    def end(self):
        self.results = dict(pos=self.pos, time=np.array(self.t)-self.t[0], guess=self.guess, heat=self.heat, contour=self.contour, pct_xadj=self.pct_xadj)
        np.savez(self.fh.make_path('tracking.npz'), **self.results)
        savemat(self.fh.make_path('tracking.mat'), self.results)
        
        self.mov.release()
        destroyAllWindows()
    def man_update(self, d):
        for k,v in d.items():
            setattr(self,k,v)
    def make_rooms(self):
        self.path_x = mpl_path.Path(self.pts[np.array([self.xmli,self.xoli,self.xori,self.xmri])])
        self.path_y = mpl_path.Path(self.pts[np.array([self.ymli,self.yoli,self.yori,self.ymri])])
        self.path_z = mpl_path.Path(self.pts[np.array([self.zmli,self.zoli,self.zori,self.zmri])])

        #experimental: hand in frame on x room
        self.path_x_adj = mpl_path.Path(self.pts[np.array([self.xoli,self.xoli_adj,self.xori_adj,self.xori])])
        self.xadj_mask = np.zeros((self.height,self.width))
        for iy in xrange(self.xadj_mask.shape[0]):
            for ix in xrange(self.xadj_mask.shape[1]):
                self.xadj_mask[iy,ix] = self.path_x_adj.contains_point([ix,iy])
        self.xadj_idxs = np.squeeze(np.argwhere(self.xadj_mask==True))

        self.border_mask = np.zeros((self.height,self.width))
        pthpts = self.pts[np.array([self.yoli_adj,self.yori_adj,self.ymri,self.ycri,self.zmli,self.zoli_adj,self.zori_adj,self.zmri,self.zcri,self.xmli,self.xoli_adj,self.xori_adj,self.xmri,self.xcri,self.ymli])]
        pth = mpl_path.Path(pthpts)
        for iy in xrange(self.border_mask.shape[0]):
            for ix in xrange(self.border_mask.shape[1]):
                self.border_mask[iy,ix] = pth.contains_point([ix,iy])
    def classify_pts(self):
        #stored in (x,y)
        #c: center
        #m: middle
        #o: out
        #x: bottom arm, y: left arm, z: right arm
        #l: left when going down arm, r: right when going down arm
        #pt is: [x/y/z c/m/o l/r]
        X,Y = 0,1
        def nn(pidx,n,ex=[]):
            #idxs of n closest pts to p, excluding all idxs in ex
            p = self.pts[pidx]
            ds = np.array([dist(pp,p) for pp in self.pts])
            idxs =  np.argsort(ds)
            idxs = np.array([i for i in idxs if i not in ex])
            return idxs[:n]
        def sortby(pidxs, dim):
            pts = self.pts[np.array(pidxs)]
            return pidxs[np.argsort(pts[:,dim])]
        dists = np.array([dist(self.pts_c, p) for p in self.pts])
        c3i = self.c3i[np.argsort(self.pts[self.c3i][:,0])]
        m6i = self.m6i
        o6i = self.o6i
        
        #classify them:
        xcri=ycli=c3i[0]
        ycri=zcli=c3i[1]
        zcri=xcli=c3i[2]
        temp = nn(xcri, 2, ex=c3i)
        ymli,xmri = sortby(temp, Y)
        temp = nn(ycri, 2, ex=c3i)
        ymri,zmli = sortby(temp, X)
        temp = nn(zcri, 2, ex=c3i)
        zmri,xmli = sortby(temp, Y)
        cm9 = [xcri,ycri,zcri,xmri,xmli,ymri,ymli,zmri,zmli]

        xoli = nn(xmli, 1, ex=cm9)[0]
        xori = nn(xmri, 1, ex=cm9)[0]
        yoli = nn(ymli, 1, ex=cm9)[0]
        yori = nn(ymri, 1, ex=cm9)[0]
        zoli = nn(zmli, 1, ex=cm9)[0]
        zori = nn(zmri, 1, ex=cm9)[0]
        
        #accounting for inner wall reflections:
        zol,zml = np.array([self.pts[zoli],self.pts[zmli]]).astype(np.int32)
        dd = np.sqrt(np.sum((zol-zml)**2))
        cup_dd = 0.80*dd
        #z cup:
        zol,zml = np.array([self.pts[zoli],self.pts[zmli]]).astype(np.int32)
        d_zl = zol-zml
        theta_zl = np.arctan2(*d_zl)
        l2 = zml + cup_dd * np.array([np.sin(theta_zl),np.cos(theta_zl)])
        zor,zmr = np.array([self.pts[zori],self.pts[zmri]]).astype(np.int32)
        d_zr = zor-zmr
        theta_zr = np.arctan2(*d_zr)
        r2 = zmr + cup_dd * np.array([np.sin(theta_zr),np.cos(theta_zr)])
        zr2,zl2 = r2,l2
        #y cup:
        yol,yml = np.array([self.pts[yoli],self.pts[ymli]]).astype(np.int32)
        d_yl = yol-yml
        theta_yl = np.arctan2(*d_yl)
        l2 = yml + cup_dd * np.array([np.sin(theta_yl),np.cos(theta_yl)])
        yor,ymr = np.array([self.pts[yori],self.pts[ymri]]).astype(np.int32)
        d_yr = yor-ymr
        theta_yr = np.arctan2(*d_yr)
        r2 = ymr + cup_dd * np.array([np.sin(theta_yr),np.cos(theta_yr)])
        yr2,yl2 = r2,l2
        #x cup:
        xol,xml = np.array([self.pts[xoli],self.pts[xmli]]).astype(np.int32)
        d_xl = xol-xml
        theta_xl = np.arctan2(*d_xl)
        l2 = xml + cup_dd * np.array([np.sin(theta_xl),np.cos(theta_xl)])
        xor,xmr = np.array([self.pts[xori],self.pts[xmri]]).astype(np.int32)
        d_xr = xor-xmr
        theta_xr = np.arctan2(*d_xr)
        r2 = xmr + cup_dd * np.array([np.sin(theta_xr),np.cos(theta_xr)])
        xr2,xl2 = r2,l2
        self.pts = np.rint(np.concatenate((self.pts, [zr2,zl2,yr2,yl2,xr2,xl2])))

        pts_dict = dict(pts=self.pts,xcri=xcri,ycli=ycli,ycri=ycri,zcli=zcli,zcri=zcri,xcli=xcli,xmri=xmri,ymli=ymli,ymri=ymri,zmli=zmli,xmli=xmli,zmri=zmri,xoli=xoli,xori=xori,yoli=yoli,yori=yori,zoli=zoli,zori=zori,zori_adj=-6,zoli_adj=-5,yori_adj=-4,yoli_adj=-3,xori_adj=-2,xoli_adj=-1)
        self.man_update(pts_dict)

        np.savez(self.fh.make_path('pts.npz', mode=self.fh.BL), **pts_dict)
    def verify_pts(self, add_to_all=True):
        if add_to_all:
            self.all_possible_pts += list(self.pts)
        if len(self.pts) < 15:
            return False
        if len(self.pts) > 25:
            return False
        elif len(self.pts) > 15:
            allperms = np.array(list(it.combinations(range(len(self.pts)), 15)))
            if len(allperms)>200:
                totryi = np.random.choice(range(len(allperms)),200,replace=False)
                totryi = allperms[totryi]
            else:
                totryi = allperms
        elif len(self.pts) == 15:
            totryi = [range(15)]
        
        for ptsi in totryi:
            pts = self.pts[ptsi]
            self.pts_c = np.mean(pts,axis=0)
            dists = np.array([dist(self.pts_c, p) for p in pts])
            c3i = np.argsort(dists)[:3]
            m6i = np.argsort(dists)[3:9]
            o6i = np.argsort(dists)[9:]

            good = True
            
            #test dists from center
            if np.std(dists[c3i]) > 1. or np.std(dists[m6i]) > 1. or np.std(dists[o6i]) > 2.5:
                good = False
            #x = self.background.copy()
            #for pt in pts:
            #    cv2.circle(x, tuple(pt), 4, (255,255,255), thickness=3)
            #pl.figure(2);pl.imshow(x)
            #raw_input()
            ##
            
            #test outer dists from each other
            o6 = pts[o6i]
            omindists = np.array([np.min(np.array([dist(p,pp) for p in o6])) for pp in o6])
            if np.std(omindists) > 0.6:
                good = False
           
           #test middle dists from each other
            m6 = pts[m6i]
            mmindists = np.array([np.min(np.array([dist(p,pp) for p in m6])) for pp in m6])
            if np.std(mmindists) > 0.6:
                good = False
            
            if good:
                self.c3i = c3i
                self.m6i = m6i
                self.o6i = o6i
                self.pts = pts
                return True
           
        return False
    def permute_pts(self):
        #NOT IN USE
        apts = np.array(self.all_possible_pts)
        unpts = []
        for pt in apts:
            if len(unpts)==0:
                unpts.append(pt)
            else:
                dists = np.sqrt(np.sum((pt-np.array(unpts))**2,axis=1))
                mindist = np.min(dists)
                if mindist > 5:
                    unpts.append(pt)
        unpts = np.array(unpts)
        if len(unpts)<15:
            return False
        for _ in xrange(10000):
            testpts = np.random.choice(np.arange(len(unpts)),15, replace=False)
            testpts = unpts[testpts]
            self.pts = testpts
            ##
            #x = self.background.copy()
            #for pt in self.pts:
            #    cv2.circle(x, tuple(pt), 4, (255,255,255), thickness=3)
            #pl.figure(2);pl.imshow(x)
            ##
            if self.verify_pts(add_to_all=False):
                return True
        return False
    def load_pts(self, mode='auto'):
        try:
            self.man_update(np.load(self.fh.make_path('pts.npz', mode=self.fh.BL)))
        except:
            if mode == 'auto':
                self.all_possible_pts = []
                invalid = True
                attempts = 0
                while invalid:
                    if attempts > 500:
                        raise Exception('Pts cannot be found.')
                    img = self.background_image.copy()
                    lp_ksizes = [13,15,17,19,21,23,25] #from 5-15 before
                    lp_ksize = rand.choice(lp_ksizes)
                    sbd_areas = [range(3,20), range(61,140)] #8,26 46,55
                    sbd_area = [rand.choice(sbd_areas[0]), rand.choice(sbd_areas[1])]
                    sbd_circs = [np.arange(0.05,0.35), range(1000,1001)]#0.19,0.35 1000
                    sbd_circ = [rand.choice(sbd_circs[0]), rand.choice(sbd_circs[1])]
                    subtr_rowmeans = rand.choice([True,False])

                    if subtr_rowmeans:
                        img = img-np.mean(img,axis=1)[:,None]
                    img = cv2.Laplacian(img, cv2.CV_32F, ksize=lp_ksize)
                    img += abs(img.min())
                    img = img/img.max() *255
                    img = img.astype(np.uint8)

                    #pl.figure(1);pl.imshow(img,cmap=pl.cm.Greys_r)
                    params = cv2.SimpleBlobDetector_Params()
                    params.filterByArea = True
                    params.filterByCircularity = True
                    params.minArea,params.maxArea = sbd_area
                    params.minCircularity,params.maxCircularity = sbd_circ
                    detector = cv2.SimpleBlobDetector(params)
                    fs = detector.detect(img)
                    pts = np.array([f.pt for f in fs])
                    pts = np.round(pts).astype(np.uint32)
                    x = img.copy()
                    for pt in pts:
                        cv2.circle(x, tuple(pt), 4, (255,255,255), thickness=3)
                    #pl.figure(2);pl.imshow(x);raw_input()
                    self.pts = pts
                    invalid = not self.verify_pts()
                    attempts += 1
            elif mode == 'manual':
                pl.imshow(self.background_image, cmap=pl.cm.Greys_r)
                pl.title('center3->middle6->outer6')
                pts = pl.ginput(n=15, timeout=-1)
                pts = np.round(pts).astype(np.uint32)
                self.pts = np.array(pts)
                pl.close()
                # note that verify is being skipped
                self.c3i,self.m6i,self.o6i = np.arange(0,3),np.arange(3,9),np.arange(9,15)
                self.pts_c = np.mean(self.pts, axis=0)
            self.classify_pts()
    def load_time(self):
        with open(self.fh.get_path(self.fh.TRIAL,self.fh.TIME),'r') as f:
            self.time = np.array(json.loads(f.read()))
        self.Ts = np.mean(self.time[1:]-self.time[:-1])
        self.fs = 1/self.Ts
    def load_background(self):
        try:
            bg = np.load(self.fh.make_path('background.npz',mode=self.fh.BL))
            background = bg['computations']
            background_image = bg['image']
        except:
            blmov = VideoCapture(self.fh.get_path(self.fh.BL,self.fh.MOV))
            valid, background, ts = self.get_frame(blmov, n=-1, blur=True)
            blmov.release()
            
            blmov = VideoCapture(self.fh.get_path(self.fh.BL,self.fh.MOV))
            valid, background_image, ts = self.get_frame(blmov, n=-1, blur=False)
            blmov.release()
            
            np.savez(self.fh.make_path('background.npz',mode=self.fh.BL), computations=background, image=background_image)
        self.background, self.background_image = background, background_image
    def get_frame(self, mov, n=1, skip=0, blur=True):
        for s in range(skip):
            mov.read()
            self.framei += 1 #the number of frames that have been read
        if n==-1:
            n = 99999999999999999.
        def get():
            valid, frame = mov.read()
            if not valid:
                return (False, None, None)
            ts = self.time[self.framei]
            self.framei += 1
            frame = frame.astype(np.float32)
            frame = cvtColor(frame, CV_RGB2GRAY)
            if blur:
                frame = GaussianBlur(frame, (self.kernel,self.kernel), 0)
            return valid,frame,ts

        valid,frame,ts = get()
        i = 1
        while valid and i<n:
            valid,new,ts = get()
            i += 1
            if valid:
                frame += new
        
        if frame is not None:
            frame = frame/i
        return (valid, frame, ts)
    def find_possible_contours(self, frame, consecutive_skips):
        self.diff = absdiff(frame,self.background)
        _, self.diff = threshold(self.diff, self.diff_thresh, 1, THRESH_BINARY)
        diff_raw = self.diff.copy()
        self.diff = self.diff*self.border_mask
        edges = Canny(self.diff.astype(np.uint8), self.cth1, self.cth2)
        contours, hier = findContours(edges, RETR_EXTERNAL, CHAIN_APPROX_TC89_L1)
        #contours = [c for c in contours if not any([pa.contains_point(contour_center(c)) for pa in self.paths_ignore])]
        if consecutive_skips>self.consecutive_skip_threshold:
            consecutive_skips=0
            possible = contours
        else:
            possible = [c for c in contours if dist(contour_center(c),self.last_center)<self.translation_max]
        return possible, diff_raw
    def choose_best_contour(self, possible):
        chosen = possible[np.argmax([cv2.arcLength(c,False) for c in possible])] #arcLength or contourArea
        center = contour_center(chosen,asint=True)[0]
        return chosen,center
    def label_frame(self, frame, center):
        showimg = np.copy(frame).astype(np.uint8)
        if self.path_x.contains_point(center):
            color = (0,0,0)
        elif self.path_y.contains_point(center):
            color = (210,210,210)
        elif self.path_z.contains_point(center):
            color = (100,100,100)
        else:
            color = (255,255,255)
        circle(showimg, tuple(center), radius=10, thickness=5, color=color)
        for pt in self.pts.astype(int):
            circle(showimg, tuple(pt), radius=4, thickness=3, color=(0,0,0))
        return showimg
    def show_frame(self, frame, wait=1):
        cv2imshow('Tracking',frame)
        waitKey(wait)
    def run(self, show=False, save=False, tk_var_frame=None, wait=1, start_pos='none'):
        #interfaces
        if show or save:
            fsize = (self.width*2, self.height)
            save_frame = np.zeros([self.height, self.width*2, 3], dtype=np.uint8)
        if show:
            namedWindow('Tracking')
        if save:
            writer = VideoWriter()
            writer.open(self.fh.make_path('tracking.avi'),self.fourcc,round(self.fs),frameSize=fsize,isColor=True)
        
        #run
        self.framei = 0
        self.pos = []
        self.t = []
        self.guess = []
        self.contour = []
        self.pct_xadj = []
        self.heat = np.zeros((self.height,self.width))
        consecutive_skips = 0
        if start_pos == 'x':
            start_pts = [self.xori_adj, self.xoli_adj]
        elif start_pos == 'y':
            start_pts = [self.yori_adj, self.yoli_adj]
        elif start_pos == 'z':
            start_pts = [self.zori_adj, self.zoli_adj]
        elif start_pos == 'c':
            start_pts = [self.zori, self.xori, self.yori]
        elif start_pos == 'none':
            start_pts = [self.zori, self.xori, self.yori]
            consecutive_skips = self.consecutive_skip_threshold+1
        self.last_center = np.mean(self.pts[np.array(start_pts)],axis=0).astype(int)
        self.last_contour = np.array([self.last_center])
        valid,frame,ts = self.get_frame(self.mov,skip=self.resample-1)
        while valid:
            possible,diff_raw = self.find_possible_contours(frame,consecutive_skips)
            self.pct_xadj.append(np.mean( diff_raw[self.xadj_idxs[:,0],self.xadj_idxs[:,1]]))
            
            if len(possible) == 0:
                center = self.last_center
                contour = self.last_contour
                self.guess.append(True)
                consecutive_skips+=1
            else:
                contour,center = self.choose_best_contour(possible)
                self.guess.append(False)
                consecutive_skips = 0
            self.pos.append(center)
            self.contour.append(contour)
            self.t.append(ts)
            self.heat[center[1],center[0]] += 1
            
            if show or save:
                lframe = self.label_frame(frame, center)
                save_frame[:,:self.width, :] = cvtColor(lframe.astype(np.float32), CV_GRAY2RGB)
                save_frame[:,self.width:, :] = cvtColor((self.diff*255).astype(np.float32), CV_GRAY2RGB)
            if show:
                self.show_frame(save_frame, wait=wait)
            if save:
                writer.write(save_frame)
             
            self.last_center = center
            self.last_contour = contour
            valid,frame,ts = self.get_frame(self.mov,skip=self.resample-1)
        
            if tk_var_frame != None:
                tk_var_frame[0].set('%i/%i'%(self.results['n_frames'], len(self.time)/float(self.resample) ))
                tk_var_frame[1].update()
        if save:
            writer.release()
        self.end()


class MainFrame(object):
    def __init__(self, parent, options=[]):
        self.parent = parent
        self.frame = tk.Frame(self.parent)
        
        self.selection = None
        self.show = tk.IntVar()
        self.save = tk.IntVar()
        self.resample = tk.StringVar()
        self.resample.set('5')
        self.diff_thresh = tk.StringVar()
        self.diff_thresh.set('80')
        
        self.initUI(options)
    def initUI(self, options):
        self.parent.title('Select Mouse')

        self.lb = tk.Listbox(self.frame, selectmode=tk.EXTENDED, exportselection=0)
        self.lb2 = tk.Listbox(self.frame, selectmode=tk.MULTIPLE, exportselection=0)
        for i in options:
            self.lb.insert(tk.END, i)
            self.lb2.insert(tk.END, i)
        self.lb2.select_set(0, len(options)-1)

        self.ok = ttk.Button(self.frame, text='OK')
        self.ok.bind("<Button-1>", self.done_select)
        self.ok.bind("<Return>", self.done_select)

        self.show_widg = ttk.Checkbutton(self.frame, text='Show tracking', variable=self.show)
        self.save_widg = ttk.Checkbutton(self.frame, text='Save tracking video', variable=self.save)
        self.resample_widg = tk.Entry(self.frame, textvariable=self.resample, state='normal')
        self.diff_thresh_widg = tk.Entry(self.frame, textvariable=self.diff_thresh, state='normal')
        label1 = ttk.Label(self.frame, text='Resample:')
        label2 = ttk.Label(self.frame, text='Threshold:')

        ttk.Label(self.frame, text="Mice to process:").grid(row=0, column=0)
        self.lb.grid(row=1, column=0)
        ttk.Label(self.frame, text="Attempt selections from:").grid(row=0, column=1)
        self.lb2.grid(row=1, column=1)
        label1.grid(row=2, column=0)
        label2.grid(row=3, column=0)
        self.resample_widg.grid(row=2, column=1)
        self.diff_thresh_widg.grid(row=3, column=1)
        self.save_widg.grid(row=4, column=1)
        self.show_widg.grid(row=4, column=0)
        self.ok.grid(row=6)
        
        self.frame.pack(fill=tk.BOTH, expand=1)

    def done_select(self, val):
        idxs = map(int, self.lb.curselection())
        values = [self.lb.get(idx) for idx in idxs]
        self.selection = values
        idxs2 = map(int, self.lb2.curselection())
        values2 = [self.lb2.get(idx) for idx in idxs2]
        self.selection2 = values2

        self.main()
    def main(self):
        self.parent.title('Status')
        self.frame.destroy()
        self.frame = ttk.Frame(self.parent, takefocus=True)
        self.frame.pack(fill=tk.BOTH, expand=1)
        self.todo = tk.StringVar()
        self.status1 = tk.StringVar()
        self.status2 = tk.StringVar()
        label_todo = ttk.Label(self.frame, textvariable=self.todo)
        label1 = ttk.Label(self.frame, textvariable=self.status1)
        label2 = ttk.Label(self.frame, textvariable=self.status2)
        self.todo.set('Mice:\n'+'\n'.join(self.selection))
        label2.pack(side=tk.TOP)
        label1.pack(side=tk.TOP)
        label_todo.pack(side=tk.BOTTOM)

        mice = self.selection
        select_from = self.selection2
        save_tracking_video = self.save.get()
        show_live_tracking = self.show.get()
        resample = int(self.resample.get())
        diff_thresh = int(self.diff_thresh.get())
        for mouse in mice:
            self.status1.set('Now processing mouse \'%s\'.'%(mouse))
            self.frame.update()
            try:
                mt = MouseTracker(mouse=mouse, data_dir=data_dir, resample=resample, diff_thresh=diff_thresh, selection_from=select_from)
                mt.run(show=show_live_tracking, save=save_tracking_video, tk_var_frame=(self.status2, self.frame))
                if mouse not in select_from:
                    select_from += [mouse]
            except:
                pass
        self.parent.destroy()


if __name__=='__main__':
    
    mode = 'nongui'
    
    if mode == 'gui':
        root1 = tk.Tk()
        data_dir = askdirectory(parent=root1, initialdir='~/Desktop', mustexist=True, title='Select directory containing data folders.')
        root1.destroy()
        if not data_dir:
            sys.exit(0)

        root = tk.Tk()
        root.geometry("400x400+200+100")
        
        options = [o for o in os.listdir(data_dir) if os.path.isdir(pjoin(data_dir,o))]
        
        frame = MainFrame(root, options=options)
        root.mainloop()

    elif mode == 'nongui':
        data_dir = '/Volumes/wang/abadura/Y-Maze/DREADDs/'
        data_dir = '/Users/Benson/Desktop'
        mouse = 'Black6_Y_2_hab'

        mt = MouseTracker(mouse=mouse, n=2, data_dir=data_dir, diff_thresh=95)
        mt.run(show=False, save=False, wait=1, start_pos='none')
