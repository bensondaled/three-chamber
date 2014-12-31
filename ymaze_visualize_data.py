import numpy as np
import pylab as pl
import os
pjoin = os.path.join
import csv

class Mouse(object):
    ACQ = 0
    REV = 1
    TEST = 2
    FORCE = -1
    DREADD = 1
    BL6 = 0
    def __init__(self, full):
        print "NOT DONE, REV still sketchy"
        full = full.upper()
        self.full = full
        if 'DREADD' in full:
            self.cond = self.DREADD
        elif 'Black6' in full:
            self.cond = self.BL6
        if 'CQ' in full:
            i1 = full.index('CQ')
            self.name = full[:i1-2].replace('_','')
            self.phase = self.ACQ
            self.day = 0
            self.n = int(full[-1])
        elif 'REV' in self.full:
            i1 = full.index('REV')
            self.name = full[:i1-1].replace('_','')
            self.phase = self.REV
            self.day = int(full[i1+4])
            self.n = full[-1]
            if self.n == 'R':
                self.n = self.FORCE
            self.n = int(self.n)
        elif 'TEST' in self.full:
            i1 = full.index('TEST')
            self.name = full[:i1-1].replace('_','')
            self.phase = self.TEST
            self.day,self.n = 0,0
        else:
            raise Exception('Cannot parse %s'%full)
    def __str__(self):
        return "%s -- %s -- day%i -- #%i"%(self.name,self.phase,self.day,self.n)

data_dir = '/Users/Benson/Desktop'
summ_file = pjoin(data_dir, 'summary.csv')
with open(summ_file,'r') as f:
    data = csv.DictReader(f)
    data = list(data)
    names = np.unique([Mouse(d['mouse']).name for d in data])
    print names
