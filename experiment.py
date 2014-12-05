from monitor import Monitor
from cameras import Camera, BW, COLOR
import numpy as np
import os
import sys
import time

if __name__=='__main__':
    name = None
    while name==None or name in os.listdir('.'):
        name = raw_input('Enter name of experiment (will default to date string):')
        if name in os.listdir('.'):
            print "Experiment file with this name already exists."
    
    duration = None
    while type(duration) != float:
        try:
            duration = float(raw_input('Enter duration of experiment in seconds:'))
        except:
            pass
    print "Please wait while baseline is acquired."
    sys.stdout.flush()
    dirr = os.path.join('data',name)
    
    #baseline
    cam = Camera(0, frame_rate=50, resolution=(640,480), color_mode=COLOR)
    mon = Monitor(cam, show=True, run_name=name+'_BL', duration=10., dirr=dirr)
    mon.go()
    
    _ = raw_input('Insert mouse, then hit enter.')
    
    i = 1
    cont = True
    while cont!='q':
        #test
        cam = Camera(0, frame_rate=50, resolution=(640,480), color_mode=COLOR)
        mon = Monitor(cam, show=True, run_name=name+"_%02d_"%i), duration=duration, dirr=dirr)
        mon.go()
        cont = raw_input('Run again? (Hit enter to run, type \'q\' + Enter to quit.)')
        i+=1
