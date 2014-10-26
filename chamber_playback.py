import json
import cv2
import numpy as np
import os
import csv
import tkFileDialog as tk

dirr = tk.askdirectory()
cond = os.path.split(dirr)[-1]

t = json.loads(open(os.path.join(dirr,cond+'-timestamps.json')).read())
t = np.squeeze(t)
Ts = int(np.rint(np.mean(t[1:]-t[:-1])*1000))

mov = cv2.VideoCapture(os.path.join(dirr,cond+'-cam0.avi'))

valid = True
t0 = t[0]
tidx = 0
delay = Ts
cv2.namedWindow('Movie')
cv2.createTrackbar('Ts', 'Movie', delay, 200, lambda x: x)
paused = False
starts = []
stops = []
started = False
while valid:
    if not paused:
        valid,frame = mov.read()
        tstr = "%0.3f"%(t[tidx]-t0)
        cv2.putText(frame, tstr, (0,frame.shape[0]-3), cv2.FONT_HERSHEY_SIMPLEX, 1., (255,255,255))
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
            stops.append(-1)
        starts.append(t[tidx-1])
        started = True
    elif k == ord('m'):
        if not started:
            starts.append(-1)
        stops.append(t[tidx-1])
        started = False

if len(starts) > len(stops):
    stops.append(-1)

cv2.destroyAllWindows()
fw = open(os.path.join(dirr,'results-'+cond+'.csv'), 'w')
dw = csv.DictWriter(fw, fieldnames=['start','stop'])
dw.writeheader()
data = [dict([('start', st),('stop', sto)]) for st,sto in zip(starts,stops)]
for d in data:
    dw.writerow(d)
fw.close()
