import sys
import numpy as np
import numpy.lib.recfunctions as nprf
from sklearn import linear_model
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['figure.facecolor'] = (1,1,1,1)

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def histplot(X,n=10, label=None, log=True):
    if n=='auto':
        n = len(X)/1000
    
    increment = (X.max() - X.min())/n
    i = X.min()
    maxi = X.max()

    x = []
    y = []
    while i < maxi:
        xcount = np.sum((X>=i) & (X<(i+increment)))
        i += increment
        
        if xcount==0: continue
            
        if log: xbin = np.log10(xcount/(increment*3*1000**3))
        else: xbin = xcount
        
        x.append(i-increment/2)
        y.append(xbin)

    plt.plot(x, 
             y,
             label=label
            )

    return x,y

def binnedplot(X,Y, n=10, percentiles = [15,50,85], mean=True, label='', ax = None):
    if n=='auto':
        n = len(X)/1000

    increment = (X.max() - X.min())/n
    i = X.min()

    x = []
    y = []
    
    y_percent = np.ndarray((0,len(percentiles)))
    
    err = []
    while i < X.max():
        bindata = Y[(X>=i) & (X<(i+increment))]
        
        i+=increment
        
        if len(bindata) == 0: continue
            
        x.append(i-increment/2)
        
        y.append(bindata.mean())
        
        y_p = np.percentile(bindata,percentiles)
        y_percent = np.append(y_percent, [y_p],axis=0)
        
        err.append(bindata.std())

        
    if mean: 
        if ax is None: plt.plot(x,y, label=label+'mean')
        else: ax.plot(x,y, label=label+'mean')
    y_percent = np.swapaxes(y_percent,0,1)
    
    for i in range(len(percentiles)):
        if ax is None:
            plt.plot(x, y_percent[i],label=label+str(percentiles[i]))
        else:
            ax.plot(x, y_percent[i],label=label+str(percentiles[i]))

    return x,y,err

def savefig(f, name, wdir, figsize=(3,3), xlabel='',ylabel='', title='', fontsize=11, tight = True):
    
    f.set_size_inches(*figsize)
    
    if xlabel!='': plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel!='': plt.ylabel(ylabel, fontsize=fontsize)
    if title!='': plt.title(title, fontsize=fontsize)
    
    if tight: plt.tight_layout()
    
    f.savefig(wdir+'images/'+name+'.pdf')
    


def get_vsig(dat):
#     return np.array([np.sqrt(np.sum(dat[i]['vlos'][:dat[i]['Ngal']]**2)/dat[i]['Ngal'])
#                  for i in range(len(dat))
#                 ])
    return np.array([np.std(dat[i]['vlos'][0:dat[i]['Ngal']])
                     for i in range(len(dat))
                    ])

def get_rsig(dat):
    return np.array([np.sqrt(np.sum(dat[i]['Rproj'][:dat[i]['Ngal']]**2)/dat[i]['Ngal'])
                     for i in range(len(dat))
                    ])
    
def add_vsig(dat):
    vsig = get_vsig(dat)
    
    if 'vsig' in dat.dtype.names:
        
        dat['vsig'] = vsig
        return dat
    
    return nprf.rec_append_fields(dat,'vsig',vsig,dtypes='<f4')

def add_rsig(dat):
    
    rsig = get_rsig(dat)
    
    if 'rsig' in dat.dtype.names:
        
        dat['rsig'] = rsig
        return dat
    
    return nprf.rec_append_fields(dat,'rsig',rsig,dtypes='<f4')

def get_massv(dat):

    numHalos = len(dat)
    numTrain = np.sum(dat['intrain'] == True)
    numTest = np.sum(dat['intest'] == True)

    featuresList_train = []
    massList_train = np.zeros(numTrain,'f')
    vsigList_train = np.zeros(numTrain,'f')

    featuresList_test = []
    massList_test = np.zeros(numTest,'f')
    vsigList_test = np.zeros(numTest,'f')

    te = 0
    tr=0

    #loop through all of the halos
    for h in range(numHalos):
        if dat['Mtot'][h]==0:
            print 'zero!'
            continue

        numSubs = dat['Ngal'][h]
        subfeat = np.zeros((numSubs, 2), 'f')
        #loop through the subhalos/galaxies in this halo:
        for sh in range (0, numSubs):
            #and fill in each of the features
            subfeat[sh][0] = np.abs(dat['vlos'][h][sh])
            subfeat[sh][1] = 1.0

        if (dat['intrain'][h]==True):
            featuresList_train.append(subfeat)
            massList_train[tr] = np.log10(dat['Mtot'][h])
            vsigList_train[tr] = np.log10(np.std(dat['vlos'][h][0:numSubs]))
            tr+=1

        if (dat['intest'][h]==True):

            featuresList_test.append(subfeat)
            massList_test[te] = np.log10(dat['Mtot'][h])
            vsigList_test[te] = np.log10(np.std(dat['vlos'][h][0:numSubs]))
            te+=1
            
    return massList_train, vsigList_train, massList_test, vsigList_test

def foldmassdist(dat):
    nFolds = max(dat['fold'])+1
    
    for i in range(nFolds):
        masses = dat['Mtot'][(dat['fold']==i) & (dat['intrain']==True)]
        
        histplot(np.log10(masses)+i, n=100, label='fold_'+str(i))

def pdfs(dat, mbins=6, vbins = 20,  norm=False, label = ''):
    if len(dat) == 0:
        return
    
    vmin = dat['vlos'].min()
    vmax = dat['vlos'].max()
    vinc = (vmax-vmin)/vbins
    
    min_mass = np.log10(min(dat['Mtot']))
    max_mass = np.log10(max(dat['Mtot']))
    
    minc = (max_mass - min_mass)/mbins
    
    mi = min_mass

    if label: label += '_'
    
#     while (mi < max_mass)
    for mi in (min_mass + minc*np.arange(mbins)):
#         print mi
        
        insample = (dat['Mtot'] > 10**mi ) & (dat['Mtot'] < 10**(mi+minc))
        
        vs = dat['vlos'][insample]
        ngals = dat['Ngal'][insample]
        
#         print len(vs)
        
        histvals = np.zeros(vbins)
        for j in range(len(vs)):
            clusv = vs[j][:(ngals[j])]

            if norm: clusv = clusv/clusv.std()
                
            vi = vmin
            
            for k in range(vbins):
                histvals[k] += np.sum((clusv >= (vi+k*vinc)) & (clusv<(vi + (k+1)*vinc)) )*1./ngals[j]
        
#         if norm: nvs = nvs/nvs.std()
            
#         if len(nvs)>0 :histplot(nvs,n=20,label="%.2f - %.2f" % (i,i + inc), log=False)
        plt.plot(
            vmin + 0.5*vinc + np.arange(vbins)*vinc,
            histvals/len(vs),
            label=label + "%.2f - %.2f" % (mi,mi + minc)
        )
        

        
def kde_hist(clus, label=None):
    vs = clus['vlos'][0:clus['Ngal']]
    
    kern = gaussian_kde(vs)
    pos = vs.min() + ((vs.max() -vs.min())/(2.*len(vs))) * np.arange(2*len(vs))
    y = kern(pos)

    print np.log10(clus['Mtot'])
    plt.plot(pos,y, label=label)
    
    plt.xlabel(r'$v_{los}$')
    plt.ylabel(r'$dn/dv_{los}$')
    
def parseout(dat):
    preds = np.zeros((0,3),'f')

    for i in range(len(dat)/3):
        preds = np.append(preds, [[dat[3*i], dat[3*i+1], dat[3*i+2]]],axis=0)

    return preds

def plotresult(dat,title='',ax=None):
    
    if ax is None:
        plt.plot(dat[:,1],dat[:,1],label='base')
        plt.xlabel('True log[M200c]', fontsize=18)
        plt.ylabel('SDM log[M200c]', fontsize=18)
        plt.title(title, fontsize=20)
    else:
        ax.plot(dat[:,1],dat[:,1],label='base')
#         ax.set_xlabel('True log[M200c]', fontsize=18)
#         ax.set_ylabel('SDM log[M200c]', fontsize=18)
        ax.set_title(title, fontsize=14)
        
    y = binnedplot(dat[:,1],dat[:,0],100,ax = ax)
    
    return 
    
def ploterror(dat,title='',ax=None):
    err = (dat[:,0]-dat[:,1])/dat[:,1]
    
    if ax is None:
        plt.plot(dat[:,1],[0]*len(dat),label='base')
        plt.xlabel('True log[M200c]', fontsize=18)
        plt.ylabel('SDM log[M200c]', fontsize=18)
        plt.title(title, fontsize=20)
    else:
        ax.plot(dat[:,1],[0]*len(dat),label='base')
#         ax.set_xlabel('True log[M200c]', fontsize=18)
#         ax.set_ylabel('SDM log[M200c]', fontsize=18)
        ax.set_title(title, fontsize=14)
        
    y = binnedplot(dat[:,1],err,100,ax = ax)
    
    return 
    
def plotallresults(dats):
    
    plt.plot(dats[0][1][:,1],dats[0][1][:,1],label='base')
    
    for i, dat in dats:
        y = binnedplot(dat[:,1],dat[:,0],50,percentiles=[50], mean=False,label=i)


    
def plotallerror(dats):
    
    plt.plot(dats[0][1][:,1],[0]*len(dats[0][1]),label='base')
    
    for i,dat in dats:
        err = (dat[:,0]-dat[:,1])/dat[:,1]
        y = binnedplot(dat[:,1],err,50,percentiles=[50], mean=False,label=i)


    
# def ploterror(dat,title=''):
#     err = (dat[:,0]-dat[:,1])/dat[:,1]
    
#     plt.plot(dat[:,1],[0]*len(dat),label='base')
#     y = binnedplot(dat[:,1],err,100)

#     plt.xlabel('True log[M200c]', fontsize=18)
#     plt.ylabel('SDM Fractional Mass Error', fontsize=18)
#     plt.title(title, fontsize=20)
    
#     return f