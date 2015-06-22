import json
import cv2
import numpy as np
import os
import csv
import tkFileDialog as tk
import sys

try:
    nosave = sys.argv[1]
    timestart = float(sys.argv[2])
except:
    nosave = False
    timestart = None
if nosave == 'x':
    nosave = True

dirr = tk.askdirectory(initialdir=r'Z:\abadura\Julia\DREADDs\SocialChamber\Analyzed')
cond = os.path.split(dirr)[-1]

t = json.loads(open(os.path.join(dirr,cond+'-timestamps.json')).read())
try:
    tracking = np.load(os.path.join(dirr,cond+'_tracking.npz'))
    centers_all = tracking['centers_all']
    resample = tracking['params'][-1]
except:
    tracking=None
t = np.squeeze(t)
Ts = int(np.rint(np.mean(t[1:]-t[:-1])*1000))

mov = cv2.VideoCapture(os.path.join(dirr,cond+'-cam.avi'))

valid = True
t0 = t[0]
tidx = 1
if timestart:
    tidx = np.argmin(np.abs(timestart-(t-t0)))
    for _ in xrange(tidx-1):
        mov.grab()
delay = Ts
cv2.namedWindow('Movie')
cv2.createTrackbar('Ts', 'Movie', delay, 200, lambda x: x)
paused = True
starts = []
stops = []
started = False

#initial frame
valid,frame = mov.read()
tstr = "%0.3f"%(0.0)
cv2.putText(frame, tstr, (0,frame.shape[0]-3), cv2.FONT_HERSHEY_SIMPLEX, 1., (255,255,255))
cv2.imshow('Movie', frame)
while valid:
    if not paused:
        valid,frame = mov.read()
        if not valid:
            break
        tstr = "%0.3f"%(t[tidx]-t0)
        cv2.putText(frame, tstr, (0,frame.shape[0]-3), cv2.FONT_HERSHEY_SIMPLEX, 1., (255,255,255))
        #tracking
        if tracking!=None:
            cv2.circle(frame, tuple(centers_all[max(0,(tidx-1)/resample - resample-1)]), 10, (200,200,200), thickness=3)
        cv2.imshow('Movie', frame)
        delay = cv2.getTrackbarPos('Ts', 'Movie')
        if delay == 0:
            delay = 1
        tidx += 1
    k = cv2.waitKey(delay)
    if k == ord('p'):
        paused = not paused
    elif k == ord('q'):
        break
    elif k == ord('z'):
        if started:
            stops.append('(none)')
        starts.append(tstr)
        started = True
    elif k == ord('m'):
        if not started:
            starts.append('(none)')
        stops.append(tstr)
        started = False

if len(starts) > len(stops):
    stops.append('(none)')

cv2.destroyAllWindows()
if not nosave:
    fw = open(os.path.join(dirr,'results-'+cond+'.csv'), 'w')
    dw = csv.DictWriter(fw, fieldnames=['start','stop'])
    dw.writeheader()
    data = [dict([('start', st),('stop', sto)]) for st,sto in zip(starts,stops)]
    for d in data:
        dw.writerow(d)
    fw.close()
