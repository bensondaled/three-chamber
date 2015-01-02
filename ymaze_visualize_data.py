import numpy as np
import pylab as pl
import os
pjoin = os.path.join
import csv

class Mouse(object):
    ACQ = 0
    REV = 1
    TEST = 2
    FORCE = 3
    DREADD = 1
    BL6 = 0
    def __init__(self, full):
        self.dic_cond = {0:'BL6',1:'DREADD'}
        self.dic_phase = {0:'acquisition',1:'reversal',2:'test',3:'force'}
        full = full.upper()
        self.full = full
        if 'DREADD' in full:
            self.cond = self.DREADD
            i0 = full.index('DREADD')+6
        elif 'BLACK6' in full:
            self.cond = self.BL6
            i0 = full.index('BLACK6')+6
        if 'CQ' in full:
            i1 = full.index('CQ')
            self.phase = self.ACQ
            self.day = 0
            self.n = int(full[-1])
        elif 'REV' in self.full:
            i1 = full.index('REV')
            self.phase = self.REV
            if self.cond == self.DREADD:
                self.day = int(full[i1+4])
            elif self.cond == self.BL6:
                self.day = int(full[i1+3])
            self.n = full[-1]
            if 'FOR' in self.full:
                self.phase = self.FORCE
                self.n = -1
            self.n = int(self.n)
        elif 'FOR' in self.full:
            i1 = full.index('FOR')
            self.phase = self.FORCE
            self.day,self.n = 0,0
        elif 'TEST' in self.full:
            i1 = full.index('TEST')
            self.phase = self.TEST
            self.day,self.n = 0,0
        else:
            raise Exception('Cannot parse %s'%full)
        self.name = full[i0:i1-1].replace('_','')
    def __str__(self):
        return "%s -- %s -- day%i -- #%i"%(self.name,self.phase,self.day,self.n)

data_dir = '/Users/Benson/Desktop'

summ_file = pjoin(data_dir, 'black6.csv');pl.figure(1)
#summ_file = pjoin(data_dir, 'dreadds.csv');pl.figure(2)

with open(summ_file,'r') as f:
    data = csv.DictReader(f)
    data = list(data)
    mice = [Mouse(d['mouse']) for d in data]
    data = [(m.dic_cond[m.cond],m.name,m.full,m.dic_phase[m.phase],m.day,m.n,d['n'],d['score'],d['time_to_correct'],d['distance']) for d,m in zip(data,mice)]
    data = np.array(data, dtype=[('cond','a6'),('id','a5'),('full','a50'),('phase','a11'),('day',int),('session_n',int),('trial_n',int),('score','a9'),('ttc',float),('dist',float)])
    data = data.view(np.recarray)
   
    dic_score = dict(correct=1,incorrect=0,null=0,none=0)
    for idx,mouse in enumerate(np.unique(data.id)):
        rows = data[data.id == mouse]
        rows = rows[rows.cond != 'force']
        avgs = [np.mean([dic_score[d['score']] for d in rows[rows.full == f]]) for f in np.unique(rows.full)]
        pl.subplot(2,3,idx+1)
        pl.plot(avgs, '-o')
        pl.ylim(-0.1,1.1)
