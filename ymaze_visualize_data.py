import numpy as np
import pylab as pl
import os
pjoin = os.path.join
import csv

def safemean(l):
    if len(l):
        return np.nanmean(l)
    else:
        return np.nan
def allmean(l):
    try:
        res = np.mean(l)
    except:
        res = l[0]
    return res
def sem(l, axis=0):
    ns = [len(i[np.logical_not(np.isnan(i))]) for i in l.T]
    return np.nanstd(l,axis=axis)/np.sqrt(ns)
def seme(l, axis=0):
    if np.any(np.isnan(l)):
        print "ERROR!"
    n = l.shape[0]
    mad =  np.median(np.abs(l - np.median(l, axis)), axis)
    return 1.4826*mad / n

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
        self.dic_phase_short = {0:'acq',1:'rev',2:'test',3:'force'}
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
                self.n = 0
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

data_dir = '/Users/Benson/Documents/MD-PhD/PhD/aleksandra/ymaze'
forsam = []
for statmode in ['mean','median']:

    fsize=((12,9))
    pl.figure( figsize=(12,8))
    summ_file1 = pjoin(data_dir, 'black6.csv')
    summ_file2 = pjoin(data_dir, 'dreadds.csv')
    dic_score = dict(correct=1,incorrect=0,null=0,none=0)
    ignore_score = []#['null','none']

    results = [[],[]]

    for sfi,summ_file in enumerate([summ_file1,summ_file2]):

        with open(summ_file,'r') as f:
            data = csv.DictReader(f)
            data = list(data)
            #custom:
            for i,d in enumerate(data):
                if d['mouse'] =='DREADD_GR3_M4_acq4_tr05':
                    d['n'] = 5
                    d['mouse'] = 'DREADD_GR3_M4_acq4'
                if statmode=='median' and d['time_to_correct'] == '-1': #median
                    d['time_to_correct'] = 99.9
                data[i] = d
            #end custom
            mice = [Mouse(d['mouse']) for d in data]
            data = [(m.dic_cond[m.cond],m.name,m.full,m.shortcond,m.shortname,m.dic_phase[m.phase],m.day,m.n,d['n'],d['score'],d['time_to_correct'],d['distance']) for d,m in zip(data,mice)]
            data = np.array(data, dtype=[('cond','a6'),('id','a5'),('full','a50'),('shortcond','a50'),('shortname','a50'),('phase','a11'),('day',int),('session_n',int),('trial_n',int),('score','a9'),('ttc',float),('dist',float)])
            data = data.view(np.recarray)
           
        for idx,mouse in enumerate(np.unique(data.id)):
            rows = data[data.id == mouse]
            scores,phases,days,ttc,ns,dists,shorts = [],[],[],[],[],[],[]
            for ses in np.unique(rows.full):
                all_5 = rows[rows.full == ses]
                scores.append( np.mean([dic_score[d['score']] for d in all_5 if d['score'] not in ignore_score]) )
                phases.append(all_5[0]['phase'])
                days.append(all_5[0]['day'])
                ns.append(all_5[0]['session_n'])
                if statmode == 'median':
                    ttc.append(np.median([d['ttc'] for d in all_5]))
                elif statmode == 'mean':
                    ttc.append(safemean([d['ttc'] for d in all_5 if d['ttc']!=-1]))
                dists.append( safemean([d['dist'] for d in all_5]) )
                shorts.append(all_5[0]['shortcond'])
            stuff = np.array(map(np.array, [scores,phases,days,ttc,ns,dists,shorts]))
            scores,phases,days,ttc,ns,dists,shorts = stuff
            phs_1 = dict(acquisition=0,test=1,reversal=2,force=2)
            phs_2 = dict(acquisition=0,test=1,reversal=2,force=3)
            phs_1 = np.array([phs_1[i] for i in phases])
            phs_2 = np.array([phs_2[i] for i in phases])
            order = np.lexsort([ns,phs_2,days,phs_1])
            if sfi == 0:
                fi = np.argwhere(['force' in i for i in phases])[0]
                ri = np.argwhere(['rev-2.3' in i for i in shorts])[0]
                order = order[order!=fi]
                order = np.insert(order, np.argwhere(order==ri)[0], fi)
            scores,phases,days,ttc,ns,dists,shorts = [d[order] for d in stuff]
            #print mouse
            #print scores, phases, days, ttc, ns, dists, shorts
            results[sfi].append( [scores, ttc, dists, mouse, shorts])
    results = np.array(results)

    titles = ['Success Rate','Time to Correct Cup (s)','Distance']
    conds = ['Black6', 'DREADDs']
    for i in xrange(3):
        pl.subplot(3,1,i+1)
        dat0 = np.array([k[i] for k in results[0]],dtype=float)
        dat1 = np.array([k[i] for k in results[1]],dtype=float)
        if statmode == 'mean':
            me0,se0 = np.nanmean(dat0,axis=0),sem(dat0,axis=0)
            me1,se1 = np.nanmean(dat1,axis=0),sem(dat1,axis=0)
        elif statmode == 'median':
            me0,se0 = np.median(dat0,axis=0),seme(dat0,axis=0)
            me1,se1 = np.median(dat1,axis=0),seme(dat1,axis=0)
        pl.errorbar(range(len(me0)),me0,se0, fmt='-o', label='Black6')
        pl.errorbar(range(len(me1)),me1,se1, fmt='-o', label='DREADDs')
        if i==0:
            pl.legend(loc='best')
        pl.ylabel(titles[i])
        pl.xticks(range(len(results[1][0][i])), results[1][0][4], rotation=30, ha='right')
        if i==0:
            pl.ylim(0,1.05)
        pl.xlim(-0.1, len(results[1][0][0])-0.9)
    pl.gcf().set_tight_layout(True)

    #for sam:
    for cond in results:
        for mo in cond:
            dic = dict(name=mo[3], ttc=mo[1], statmode=statmode)
            forsam.append(dic)

means = forsam[:10]
medians = forsam[10:]
mice = zip(means,medians)







            #wkg = zip(*[[np.mean([dic_score[d['score']] for d in rows[rows.full == f]]),rows[np.argwhere(rows.full==f)[0]].shortcond[0],rows[np.argwhere(rows.full==f)[0]].phase[0],rows[np.argwhere(rows.full==f)[0]].day[0],safemean([d['ttc'] for d in rows[rows.full == f] if d['ttc']!=-1 ]),safemean([d['dist'] for d in rows[rows.full == f]])] for f in np.unique(rows.full)])
            #avgs,fs,phs,day,ttc,dist = map(np.array,wkg)
            #
            #pl.figure(1, figsize=fsize)
            #pl.subplot(5,2, (sfi+1) + 2*(idx))
            #pl.plot(avgs[np.lexsort([fs,phs_2,day,phs_1])], '-o')
            #pl.title(rows[np.argwhere(rows.id==mouse)[0]].shortname[0])
            #pl.gca().set_ylabel('Success Rate')
            #pl.ylim(-0.1,1.1)
            #pl.xticks(range(len(avgs)), fs[np.lexsort([fs,phs_2,day,phs_1])], rotation=30, ha='right')
            #pl.xlim(-0.2,len(avgs)-0.8)
            #pl.gcf().set_tight_layout(True)
            #pl.savefig(pjoin(data_dir, 'success.png'))


            #pl.figure(2, figsize=fsize)
            #pl.subplot(5,2, (sfi+1) + 2*(idx))
            #pl.plot(ttc[np.lexsort([fs,phs_2,day,phs_1])], '-o')
            #pl.title(rows[np.argwhere(rows.id==mouse)[0]].shortname[0])
            #pl.gca().set_ylabel('Time to Cup (s)')
            #pl.ylim(-0.1,25)
            #pl.xticks(range(len(ttc)), fs[np.lexsort([fs,phs_2,day,phs_1])], rotation=30, ha='right')
            #pl.xlim(-0.2,len(avgs)-0.8)
            #pl.gcf().set_tight_layout(True)
            #pl.savefig(pjoin(data_dir, 'time_to_cup.png'))


            #pl.figure(3, figsize=fsize)
            #pl.subplot(5,2, (sfi+1) + 2*(idx))
            #pl.plot(dist[np.lexsort([fs,phs_2,day,phs_1])], '-o')
            #pl.title(rows[np.argwhere(rows.id==mouse)[0]].shortname[0])
            #pl.gca().set_ylabel('Distance')
            #pl.ylim(-0.1,2500)
            #pl.xticks(range(len(dist)), fs[np.lexsort([fs,phs_2,day,phs_1])], rotation=30, ha='right')
            #pl.xlim(-0.2,len(avgs)-0.8)
            #pl.gcf().set_tight_layout(True)
            #pl.savefig(pjoin(data_dir, 'distance.png'))
