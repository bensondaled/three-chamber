# For the setup implemented with the Y-maze. baseline now exist in same directory, etc.

#natives
import sys
import os
pjoin = os.path.join
import json
import re
import itertools as it

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
ion()

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
def contour_center(c):
    return np.round(np.mean(c[:,0,:],axis=0)).astype(int)
def dist(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

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
    def get_path(self, blm, mode):
        #mode is movie/timestamps/metadata
        #n is trial number, 0 refers to baseline
        filename = self.mouse
        if blm == self.BL:
            filename += '_BL'
        elif blm == self.TRIAL:
            filename += "_%02d_"%(self.n)
        if mode == self.MOV:
            filename += '-cam.avi'
        elif mode == self.TIME:
            filename += '-timestamps.json'
        elif mode == self.DATA:
            filename += '-metadata.json'
        pth = pjoin(self.dirr, filename)
        return pth
    def make_path(self, st, mode=1):
        filename = self.mouse
        if mode == self.BL:
            filename += '_'
        elif mode == self.TRIAL:
            filename += "_%02d_"%(self.n)
        pth = pjoin(self.dirr, filename+st)
        return pth

class MouseTracker(object):
    def __init__(self, mouse, n=1, data_dir='.', diff_thresh=80, resample=3, translation_max=100, smoothing_kernel=19, consecutive_skip_threshold=2, selection_from=[]):
        self.mouse = mouse
        self.n = n
        self.data_dir = data_dir
        
        # Parameters (you may vary)
        self.diff_thresh = diff_thresh
        self.resample = resample
        self.translation_max = translation_max
        self.kernel = smoothing_kernel
        self.consecutive_skip_threshold = (37./self.resample) * consecutive_skip_threshold

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

        self.load_background()
        self.load_time()
        self.height, self.width = self.background.shape
        self.mov = VideoCapture(self.fh.get_path(self.fh.TRIAL, self.fh.MOV))
        self.load_pts()
        self.classify_pts()
        self.results = dict(centers=[], heat=np.zeros((self.height,self.width)), skipped=0, centers_all=[], n_frames=0)

    def end(self):
        np.savez(self.fh.make_path('tracking.npz'), **self.results)
        savemat(self.fh.make_path('tracking.mat'), self.results)
        
        self.mov.release()
        destroyAllWindows()
    def classify_pts(self):
        glob_cent = np.mean(self.pts,axis=0)
        print glob_cent
        sys.exit(0)
        
    def load_pts(self):
        self.border_mask = np.zeros((self.height,self.width),dtype=bool)
        self.border_mask[50:450,100:550] = True
        nbins = 10
        img = self.background_image.copy()
        img2 = img.copy()
        #img *= self.border_mask
        img = cv2.Laplacian(img, cv2.CV_32F, ksize=9)
        #pl.figure();pl.imshow(img);pl.title('Laplace')
        binsy,binsx = [np.linspace(0,dim,nbins) for dim in img2.shape]
        for b1y,b0y in zip(binsy[1:],binsy[:-1]):
            for b1x,b0x in zip(binsx[1:],binsx[:-1]):
                sub = img2[b0y:b1y, b0x:b1x]
                sub[sub<np.percentile(sub,98)] = 0
                img2[b0y:b1y, b0x:b1x] = sub
        img[img2==0] = 0
        #pl.figure();pl.imshow(img);pl.title('Thresholding')
        img += abs(img.min())
        img[img>np.percentile(img,0.08)] = 0
        #pl.figure();pl.imshow(img);pl.title('2nd Thresholding')
        #pl.figure();pl.imshow(self.background_image);pl.title('Background')
        img[img>0] = 1
        img = img*255
        #pl.figure();pl.imshow(img);pl.title('Result')
        contours,_ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = map(cv2.contourArea, contours)
        best = np.argsort(areas)[::-1][:15]
        contours = np.array(contours)[best]
        x = np.zeros(img.shape, dtype=np.uint8)
        cv2.drawContours(x, contours, -1, (255,255,255))
        #pl.figure();pl.imshow(x);pl.title('Result contours')
        self.pts = np.array(map(contour_center, contours))
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
            valid, background = self.get_frame(blmov, n=-1, blur=True)
            blmov.release()
            
            blmov = VideoCapture(self.fh.get_path(self.fh.BL,self.fh.MOV))
            valid, background_image = self.get_frame(blmov, n=-1, blur=False)
            blmov.release()
            
            np.savez(self.fh.make_path('background.npz',mode=self.fh.BL), computations=background, image=background_image)
        self.background, self.background_image = background, background_image
    def get_frame(self, mov, n=1, skip=0, blur=True):
        for s in range(skip):
            mov.read()
        if n==-1:
            n = 99999999999999999.
        def get():
            valid, frame = mov.read()
            if not valid:
                return (False, None)
            frame = frame.astype(np.float32)
            frame = cvtColor(frame, CV_RGB2GRAY)
            if blur:
                frame = GaussianBlur(frame, (self.kernel,self.kernel), 0)
            return valid,frame

        valid,frame = get()
        i = 1
        while valid and i<n:
            valid,new = get()
            i += 1
            if valid:
                frame += new
        
        if frame!=None:
            frame = frame/i
        return (valid, frame)
    def find_possible_contours(self, frame, consecutive_skips):
        self.diff = absdiff(frame,self.background)
        _, self.diff = threshold(self.diff, self.diff_thresh, 1, THRESH_BINARY)
        self.diff = self.diff*self.border_mask
        edges = Canny(self.diff.astype(np.uint8), self.cth1, self.cth2)
        contours, hier = findContours(edges, RETR_EXTERNAL, CHAIN_APPROX_TC89_L1)
        #contours = [c for c in contours if not any([pa.contains_point(contour_center(c)) for pa in self.paths_ignore])]
        if consecutive_skips>self.consecutive_skip_threshold:
            consecutive_skips=0
            possible = contours
        else:
            possible = [c for c in contours if dist(contour_center(c),self.last_center)<self.translation_max]
        return possible
    def choose_best_contour(self, possible):
        chosen = possible[np.argmax([contourArea(c) for c in possible])]   
        center = contour_center(chosen)
        return center
    def label_frame(self, frame, center):
        showimg = np.copy(frame).astype(np.uint8)
        circle(showimg, tuple(center), radius=10, thickness=5, color=(255,255,255))
        return showimg
    def show_frame(self, frame):
        cv2imshow('Tracking',frame)
        waitKey(1)
    def run(self, show=False, save=False, tk_var_frame=None):
        
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
        consecutive_skips = 0
        self.last_center = np.round(np.array([self.height/2., self.width/2.])).astype(int)
        valid,frame = self.get_frame(self.mov,skip=self.resample-1)
        while valid:
            possible = self.find_possible_contours(frame,consecutive_skips)
            
            if len(possible) == 0:
                center = self.last_center
                self.results['skipped'] += 1
                consecutive_skips+=1
            else:
                center = self.choose_best_contour(possible)
                self.results['centers'].append(center)
                self.results['heat'][center[1],center[0]] += 1
            self.results['centers_all'].append(center) 
            
            if show or save:
                lframe = self.label_frame(frame, center)
                save_frame[:,:self.width, :] = cvtColor(lframe.astype(np.float32), CV_GRAY2RGB)
                save_frame[:,self.width:, :] = cvtColor((self.diff*255).astype(np.float32), CV_GRAY2RGB)
            if show:
                self.show_frame(save_frame)
            if save:
                writer.write(save_frame)
             
            self.results['n_frames'] += 1
            self.last_center = center
            
            valid,frame = self.get_frame(self.mov,skip=self.resample-1)
        
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
        data_dir = '/Users/Benson/Desktop/data/'
        mouse = 'test'

        mt = MouseTracker(mouse=mouse, data_dir=data_dir, diff_thresh=70)
        mt.run(show=True, save=False)
