### INSTRUCTIONS
"""
This file is a general script which is capable of analyzing the Y-maze movies.

To use the script:
    1. Edit the parameters at the start of this file (do not edit anything past the warning "DON'T EDIT FROM HERE").
    2. Open iPython. Ensure you are located in the folder where this script exists. (If you are not, running the script will fail immediately.) 

Options available to you:
    -mode: 
    (1) group mode allows you to process many mice one after another, or one mouse, all 5 videos.
    (2) single mode allows you to process a single movie.
    (3) collect mode runs through all the mice and puts the marking data into a spreadsheet.

    -actions:
    (1) "track" tracks the mouse's location and saves this data. (you only need to do this once ever for a mouse, unless you change tracking parameters).
    (2) "mark" uses the tracking data to determine the mouse's performance. (tracking must have been run in order for this to work).
    (3) "both" runs both tracking and marking sequentially.
    (4) "play" plays back the specified movie.

    -include_hab: determines whether or not to include habituation sessions (as defined by those which contain 'hab' in their names). This only applies when running mode=group and mice='all'/'ask', or when running mode=collect.

Points to avoid errors:
    -in the parameters section, specify parameters using *exactly* the options given at the end of the line 
    -data folders must be located in bucket:abadura\\Y-Maze, in either "Black6" or "DREADDs", and not subnested in any folders within that folder.
    -do not leave any files or folders inside any of the data storage areas, if they were not generated by these programs.
    -do not ever rename files whatsoever.
"""

### GENERAL PARAMETERS
condition = 'Black6' #OPTIONS: Black6 / DREADDs
mode = 'single' #OPTIONS: group / single / collect
actions = 'play' #OPTIONS:  track / mark / both / play
include_hab = False #OPTIONS: True / False
drive = 'Z:' #the drive on which wang lab bucket is mounted, ex 'Y:'

### FOR GROUP MODE
mice = ['all'] #OPTIONS: ['Black6_Y_1_acq1','Black6_Y_1_acq2'] / 'all' / 'ask'

### FOR SINGLE MODE
mouse = 'Black6_Y_1_acq1' #name of the folder containing the mouse's 5 trials
n = 1 #OPTIONS: 1 / 2 / 3 / 4 / 5 (movie number)

### TRACKING PARAMETERS
diff_thresh = 95
show = True #OPTIONS: True / False
save_video = False #OPTIONS: True / False
ms_bt_frames = 1 #milliseconds between frames when showing
resample_t = 1 #1 means no resampling

### MARKING PARAMETERS
resample_m = 1 #1 means no resampling
wall_thresh = 6 #may help find proper start time, ask ben before changing
wall_thresh_x = 0.33 #may help find proper start time, ask ben before changing
hand_thresh = 0.001 #may help find proper start time, ask ben before changing
start_range = 260 #may help find proper start time, ask ben before changing


### DON'T EDIT FROM HERE ###
#################################################################
### DON'T EDIT FROM HERE ###
from ymaze_track import MouseTracker
from ymaze_mark import Marker
from ymaze_track import FileHandler
import os
import csv
import numpy as np
import time
from tkFileDialog import askopenfilenames
import sys
import Tkinter as tk
from playback import Playback
data_dir = os.path.join(drive, 'abadura', 'Y-Maze_analyzed', condition)

logfile = open(os.path.join('logs','%s.log'%str(int(time.time()))), 'a')
print 'Log will be in %s'%('%s.log'%str(int(time.time())))
print 'Program running...'

if include_hab:
    exclude_word = 'EXEXEX'
elif not include_hab:
    exclude_word = 'hab'
if 'all' in mice or mice == 'all':
    mice = sorted([m for m in os.listdir(data_dir) if exclude_word not in m.lower() and m[0]!='.' and 'summary' not in m]) # this will run all mice of the selected condition

elif mode=='group' and ('ask' in mice or mice == 'ask'):
    root1 = tk.Tk()
    tempdir = os.path.join('.','temp')
    def clean():
        for f in os.listdir(tempdir):
            os.remove(os.path.join(tempdir,f))
        os.rmdir(tempdir)
    mice_names = sorted([m for m in os.listdir(data_dir) if exclude_word not in m.lower() and m[0]!='.' and 'summary' not in m])
    if os.path.exists(tempdir):
        clean()
    os.mkdir(tempdir)
    for mn in mice_names:
        open(os.path.join(tempdir,mn), 'a').close()
    mice = askopenfilenames(parent=root1, initialdir=tempdir, title='Select mice, DO NOT NAVIGATE FROM HERE')
    root1.destroy()
    clean()
    if not mice:
        sys.exit(0)
    if type(mice) in [str, unicode]:
        mice = mice.split(' ')
    mice = [os.path.split(m)[-1] for m in mice]
    print >>logfile, "Selected mice: " + str(mice);logfile.flush()

if mode == 'group':
    for idx,mouse in enumerate(mice):
        fh = FileHandler(data_dir, mouse, n=1)
        print '(%i/%i)'%(idx+1,len(mice))
        for tr in xrange(fh.get_n_trials()):
            try:
                if actions in ['track','both']:
                    print >>logfile, "(%i/%i) Tracking %s #%i"%(idx+1,len(mice),mouse,tr+1);logfile.flush()
                    mt = MouseTracker(mouse=mouse, n=tr+1, data_dir=data_dir, diff_thresh=diff_thresh, resample=resample_t)
                    mt.run(show=show, save=save_video, wait=ms_bt_frames)
                if actions in ['mark','both']:
                    print >>logfile, "(%i/%i) Marking %s #%i"%(idx+1,len(mice),mouse,tr+1);logfile.flush()
                    m = Marker(mouse=mouse, n=tr+1, data_dir=data_dir)
                    m.run(resample=resample_m, thresh_p_hand=hand_thresh, thresh_wall_dist=wall_thresh, start_range=start_range, thresh_wall_dist_x=wall_thresh_x)
            except:
                print >>logfile, "%s #%i failed."%(mouse,tr+1);logfile.flush()

elif mode == 'single':
    if actions in ['track','both']:
        print >>logfile, "Tracking %s #%i"%(mouse,n);logfile.flush()
        mt = MouseTracker(mouse=mouse, n=n, data_dir=data_dir, diff_thresh=diff_thresh, resample=resample_t)
        mt.run(show=show, save=save_video, wait=ms_bt_frames)
    if actions in ['mark','both']:
        print >>logfile, "Marking %s #%i"%(mouse,n);logfile.flush()
        m = Marker(mouse=mouse, n=n, data_dir=data_dir)
        m.run(resample=resample_m, thresh_p_hand=hand_thresh, thresh_wall_dist=wall_thresh, start_range=start_range, thresh_wall_dist_x=wall_thresh_x)
    if actions == 'play':
        print >>logfile, "Playing %s #%i"%(mouse,n);logfile.flush()
        pb = Playback(ymaze_path=mouse, ymaze_n=n, data_dir=data_dir)
        pb.play()
    
elif mode == 'collect':
    mice = [m for m in os.listdir(data_dir) if exclude_word not in m.lower() and m[0]!='.' and 'summary' not in m]

    rows = []
    for mouse in mice:
        print >>logfile, mouse;logfile.flush()
        fh = FileHandler(data_dir, mouse, n=1)
        for tr in xrange(fh.get_n_trials_wbehav()):
            fhm = FileHandler(data_dir, mouse, n=tr+1)
            data = np.load(fhm.make_path('behaviour.npz'))
            dic = dict(mouse=mouse, n=tr+1, score=data['score'], time_to_correct=data['time_to_correct'], distance=data['distance'], start_time=data['start_time'])
            rows.append(dic)
    
    rows = np.array(rows)
    names = np.array([r['mouse'] for r in rows])
    nums = np.array([r['n'] for r in rows])
    ordr = np.lexsort((nums,names))
    rows = rows[ordr]
    with open(os.path.join(data_dir,'summary.csv'),'w') as f:
        dw = csv.DictWriter(f,fieldnames=['mouse','n','score','time_to_correct','distance','start_time'])
        dw.writeheader()
        for row in rows:
            dw.writerow(row)
    
logfile.close()
print "Run complete."
