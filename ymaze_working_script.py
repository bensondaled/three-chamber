from ymaze_track import MouseTracker
from ymaze_mark import Marker
from ymaze_track import FileHandler
import os
import csv
import numpy as np

mode = 'collect' #group single collect

if mode == 'group':
    data_dir = 'Y:\\abadura\\Y-Maze\\Black6'
    mice = [m for m in os.listdir(data_dir) if 'hab' not in m.lower() and m[0]!='.' and 'summary' not in m]
    for mouse in mice:
        print "Processing %s"%(mouse)
        try:
            fh = FileHandler(data_dir, mouse, n=1)
            for tr in xrange(fh.get_n_trials()):
                #mt = MouseTracker(mouse=mouse, n=tr+1, data_dir=data_dir, diff_thresh=95)
                #mt.run(show=False, save=False)
                m = Marker(mouse=mouse, n=tr+1, data_dir=data_dir)
                m.run()
        except:
            print "%s failed."%(mouse)

elif mode == 'single':
    data_dir = 'Y:\\abadura\\Y-Maze\DREADDs'
    mouse = 'DREADD_GR3_M4_revD2_3'
    n = 5
    mt = MouseTracker(mouse=mouse, n=n, data_dir=data_dir, diff_thresh=100)
    mt.run(show=True, save=False, wait=1)
    #m = Marker(mouse=mouse, n=n, data_dir=data_dir)
    #m.run()
    
elif mode == 'collect':
    data_dir = 'Y:\\abadura\\Y-Maze\\Black6'
    mice = [m for m in os.listdir(data_dir) if 'hab' not in m.lower() and m[0]!='.' and 'summary' not in m]
    

    rows = []
    for mouse in mice:
        print mouse
        fh = FileHandler(data_dir, mouse, n=1)
        for tr in xrange(fh.get_n_trials_wbehav()):
            fhm = FileHandler(data_dir, mouse, n=tr+1)
            data = np.load(fhm.make_path('behaviour.npz'))
            dic = dict(mouse=mouse, n=tr+1, score=data['score'], time_to_correct=data['time_to_correct'], distance=data['distance'])
            rows.append(dic)
    
    rows = np.array(rows)
    names = np.array([r['mouse'] for r in rows])
    nums = np.array([r['n'] for r in rows])
    ordr = np.lexsort((nums,names))
    rows = rows[ordr]
    with open(os.path.join(data_dir,'summary.csv'),'w') as f:
        dw = csv.DictWriter(f,fieldnames=['mouse','n','score','time_to_correct','distance'])
        dw.writeheader()
        for row in rows:
            dw.writerow(row)