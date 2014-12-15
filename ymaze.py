from core.monitor import Monitor
from core.cameras import Camera, BW, COLOR
import numpy as np
import os
import sys
import time

if __name__=='__main__':
    name = None
    dirr = './data'
    while name==None or name in os.listdir(dirr):
        name = raw_input('Enter name of experiment (will default to date string):')
        if name in os.listdir(dirr):
            print "Experiment file with this name already exists."
    
    duration = None
    while type(duration) != float:
        try:
            duration = float(raw_input('Enter duration of experiment in seconds:'))
        except:
            pass
    dirr = os.path.join(dirr,name)

    side = None
    while side not in ['l','r']:
        side = raw_input('Enter correct side (l/r):').lower()
    
    raw_input('Hit Enter to acquire baseline.')
    cam = Camera(0, frame_rate=30, resolution=(640,480), color_mode=BW)
    for _ in xrange(40):
        cam.read()
    
    #baseline
    mon = Monitor(cam, show=True, run_name=name+'_BL', duration=15., dirr=dirr)
    mon.go()
    
    _ = raw_input('Hit enter to start recording.')
    
    i = 1
    cont = True
    while cont!='q':
        #test
        mon = Monitor(cam, show=True, run_name=name+"_%02d_"%i, duration=duration, dirr=dirr, extra={'side':side})
        mon.go()
        cont = raw_input('Run again? (Hit enter to run, type \'q\' + Enter to quit.)')
        i+=1

    cam.release()
