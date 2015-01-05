import numpy as np
import pylab as pl
import os
pjoin = os.path.join
import csv

def safemean(l):
    if len(l):
        return np.mean(l)
    else:
        return 0.

class Mouse(object):
    ACQ = 0
    REV = 1
    TEST = 2
    FORCE = 3
    DREADD = 1
    BL6 = 0
    def __init__(self, full):
        self.dic_cond = {0:'BL6',1:'DREADD'}
        self.dic_cond_short = {0:'BL6',1:'DR'}
        self.dic_phase = {0:'acquisition',1:'reversal',2:'test',3:'force'}
        self.dic_phase_short = {0:'acq',1:'rev',2:'atest',3:'force'}
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
            if self.cond == self.BL6 and self.full[-2] == 'V' and int(self.full[-1])==self.day:
                self.n = 0
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
        self.shortcond = "%s-%i.%i"%(self.dic_phase_short[self.phase],self.day,self.n)
        self.shortname = "%s-%s"%(self.dic_cond_short[self.cond],self.name)
    def __str__(self):
        return "%s -- %s -- day%i -- #%i"%(self.name,self.phase,self.day,self.n)

data_dir = '/Users/Benson/Desktop'

fsize=((12,9))
summ_file1 = pjoin(data_dir, 'black6.csv')
summ_file2 = pjoin(data_dir, 'dreadds.csv')

for sfi,summ_file in enumerate([summ_file1,summ_file2]):

    with open(summ_file,'r') as f:
        data = csv.DictReader(f)
        data = list(data)
        mice = [Mouse(d['mouse']) for d in data]
        data = [(m.dic_cond[m.cond],m.name,m.full,m.shortcond,m.shortname,m.dic_phase[m.phase],m.day,m.n,d['n'],d['score'],d['time_to_correct'],d['distance']) for d,m in zip(data,mice)]
        data = np.array(data, dtype=[('cond','a6'),('id','a5'),('full','a50'),('shortcond','a50'),('shortname','a50'),('phase','a11'),('day',int),('session_n',int),('trial_n',int),('score','a9'),('ttc',float),('dist',float)])
        data = data.view(np.recarray)
       
        dic_score = dict(correct=1,incorrect=0,null=0,none=0)
        for idx,mouse in enumerate(np.unique(data.id)):
            rows = data[data.id == mouse]
            rows = rows[rows.phase != 'force']
            #rows = rows[rows.phase != 'test']
            avgs,fs,ttc,dist = zip(*[[np.mean([dic_score[d['score']] for d in rows[rows.full == f]]),rows[np.argwhere(rows.full==f)[0]].shortcond[0],safemean([d['ttc'] for d in rows[rows.full == f] if d['ttc']!=-1 ]),safemean([d['dist'] for d in rows[rows.full == f]])] for f in np.unique(rows.full)])
            avgs,fs,ttc,dist = map(np.array,[avgs,fs,ttc,dist])
            
            pl.figure(1, figsize=fsize)
            pl.subplot(5,2, (sfi+1) + 2*(idx))
            pl.plot(avgs[np.argsort(fs)], '-o')
            pl.title(rows[np.argwhere(rows.id==mouse)[0]].shortname[0])
            pl.gca().set_ylabel('Success Rate')
            pl.ylim(-0.1,1.1)
            pl.xticks(range(len(avgs)), fs[np.argsort(list(fs))], rotation=30, ha='right')
            pl.xlim(-0.2,len(avgs)-0.8)
            pl.gcf().set_tight_layout(True)


            pl.figure(2, figsize=fsize)
            pl.subplot(5,2, (sfi+1) + 2*(idx))
            pl.plot(ttc[np.argsort(fs)], '-o')
            pl.title(rows[np.argwhere(rows.id==mouse)[0]].shortname[0])
            pl.gca().set_ylabel('Time to Cup (s)')
            pl.ylim(-0.1,25)
            pl.xticks(range(len(ttc)), fs[np.argsort(list(fs))], rotation=30, ha='right')
            pl.xlim(-0.2,len(avgs)-0.8)
            pl.gcf().set_tight_layout(True)


            pl.figure(3, figsize=fsize)
            pl.subplot(5,2, (sfi+1) + 2*(idx))
            pl.plot(dist[np.argsort(fs)], '-o')
            pl.title(rows[np.argwhere(rows.id==mouse)[0]].shortname[0])
            pl.gca().set_ylabel('Distance')
            pl.ylim(-0.1,2200)
            pl.xticks(range(len(dist)), fs[np.argsort(list(fs))], rotation=30, ha='right')
            pl.xlim(-0.2,len(avgs)-0.8)
            pl.gcf().set_tight_layout(True)
