#!/usr/bin/env python2.7

#Usage: apply SDM (Sutherland 2012) to HeCS catalog


import numpy as np
from array import array
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator import itemgetter
import sys
import sdm
import argparse
import pickle
from sklearn.cross_validation import KFold
from scipy.stats import gaussian_kde
from os import listdir

import time


matplotlib.rc('text',usetex=True)
matplotlib.rc('font',family='serif')
matplotlib.rc('lines',linewidth=3)
fontSize = 20
fontColor = 'k'

#############################################################################
pi = math.pi


###################### PARAMETERS: ###########################################################
redshift = '0.117'
#trainFile = '/share/scratch1/cmhicks/mattho/MDPL2_Maccgeq1e12_small_norecenter_z=' + redshift +'.npy' #training file
trainFile = '/share/scratch1/cmhicks/mattho/debugcat.npy'
#trainFile = '/home/mho1/data/MDPL2_Maccgeq1e12_small_norecenter_z=' + redshift + '_vfiltscale.npy'
subsample = 1

scenario = 'MLvKDE_debug' #name this run
featureType = 'MLv' #MLv, MLR, MLvR, MLvsR ... tells which features we're going to use
Nfolds = 10
divfunc = 'kl'
K = 4
sdrModel = 'NuSDR' #NuSDR, SDR

n_proc = 16 #set the number of processors to run on
seed = 8675309 #favorite integer
stand = True #standardizes features

outFolder = 'out/' #where to put the output
logFolder = 'log/'

#if no input file of unlabeled data, leave as None
unlabeledFile = None #'/home/cmhicks/VDF_HeCS/HeCS/output/HeCS_pared.npy'


def set_params(config):
    
    for param in config.keys():
        if param not in globals().keys():
            continue
            
        execstr = "global "+param+"; "+param + ' = '
             
        if type(config[param])==str:
            execstr += '"' + config[param] + '"'
        else: execstr += str(config[param])
        
        exec(execstr)

    print ('~~~ CONFIGS ~~~')
    print (config)
    
    return 

def loadHalos(halofile):
    halos = np.load(halofile)
    return halos


def setSDMModel(sdrModel, n_proc, divfunc, K):
    if sdrModel == 'NuSDR':
        print 'NuSDR'
        model = sdm.NuSDR(n_proc = n_proc, div_func = divfunc, K=K)
    elif sdrModel == 'SDR':
        model = sdm.SDR(n_proc = n_proc, div_func = divfunc, K=K)
    return model


def makeFeaturesList(halos, foldList, catname, featureType, \
                         stand=True, scaler = None):
    
    if foldList == None:
        print ''
        print 'check:  this should be used for unlabeled data and its training sample!'
        #we're passed a file, but no foldList, so we are going to use all of it
        if catname == 'train':
            ind = np.argwhere(halos['intrain'] == 1)
        elif catname == 'test':
            ind = np.arange(len(halos))

    else:
        print 'check:  this should be used for labeled data!'
        #we're passed a big file and have to figure out what is in the train and test catalog
        inFold = [True if (halos['fold'][i] in foldList) else False for i in range(len(halos))]
        if catname == 'train':
            ind = np.argwhere(inFold & halos['intrain'] == 1)
        elif catname == 'test':
            ind = np.argwhere(inFold & halos['intest'] == 1)

    


    numHalos = len(ind)

    """
    if catname == 'train':
        #whatever goes here is the thing we're predicting... in this case, log(M500):
        massList = np.ndarray.flatten(np.log10(halos['Mtot'][ind])).tolist()
    else:
        massList = [0]*len(ind)
    """

    massList = np.ndarray.flatten(np.log10(halos['Mtot'][ind])).tolist()
    
    print 'massList check:', massList[0:10]
    IDList = [i for i in halos['fold'][ind].flatten()]
    fold = np.ndarray.flatten(halos['fold'][ind]).tolist()

    print 'number of objects in ', catname, 'catalog with fold(s) = ', foldList, ':  ', len(ind)


    #note:  my old and new catalogs take slightly different forms.  To get everything to talk,
    #       I had to insert a *ton* of ".flatten()"
    if featureType == 'MLv':
        print 'MLv'
        #uses |v| as the only featuer
        featuresList = []
        for i, h in enumerate(ind):
            Ngal = halos['Ngal'][h].flatten()
            featuresList.append(zip(np.abs(halos['vlos'][h][0:Ngal].flatten()), np.ones(int(Ngal))))
            
    elif featureType == 'MLR':
        print 'MLR'
        #uses R as the only feature
        featuresList = []
        for i, h in enumerate(ind):
            Ngal = halos['Ngal'][h].flatten()
            featuresList.append(zip(np.abs(halos['Rproj'][h][0:Ngal].flatten()), np.ones(int(Ngal))))

    elif featureType == 'MLvR':
        print 'MLvR'
        #uses 2 features:  |v|, R
        featuresList = []
        for i, h in enumerate(ind):
            Ngal = halos['Ngal'][h].flatten()
            featuresList.append(zip(np.abs(halos['vlos'][h][0:Ngal].flatten()), \
                                    halos['Rproj'][h][0:Ngal].flatten()))

    elif featureType == 'MLvsR':
        print 'MLvsR'
        #uses 3 features:  |v|, |v|/sigma_v, R
        featuresList = []
        for i, h in enumerate(ind):
            Ngal = halos['Ngal'][h].flatten()
            featuresList.append(zip(np.abs(halos['vlos'][h][0:Ngal].flatten()), \
                                    np.abs(halos['vlos'][h][0:Ngal].flatten())/halos['sigmav'][h],
                                    halos['Rproj'][h][0:Ngal].flatten()))

    elif featureType =='MLvKDE':
        print 'MLvKDE'
        # uses gaussian kde of |v| as only feature
        featuresList = []
        maxv = halos['vlos'].max()
        maxNgal = halos['Ngal'].max()
        pos = (maxv/maxNgal) * (np.arange(maxNgal))
        for i, h in enumerate(ind):
            h = h[0]
            Ngal = halos['Ngal'][h]
            kern = gaussian_kde(halos['vlos'][h][:Ngal])
            
            y = kern.resample(maxNgal).flatten()
            toobig = np.argwhere(np.abs(y) > maxv)
            
            while (len(toobig) > 0):
                y = np.delete(y, toobig)
                y = np.append(y, kern.resample(len(toobig)).flatten())
                toobig = np.argwhere(np.abs(y) > maxv)
            
            y = np.abs(y)
            
            featuresList.append(zip(y, np.ones(int(maxNgal))))
            
    elif featureType == 'MLvKDEadd':
        print 'MLvKDEadd'
        featuresList = []
        maxv = halos['vlos'].max()
        maxNgal = halos['Ngal'].max()
        pos = (maxv/maxNgal) * (np.arange(maxNgal))
        for i, h in enumerate(ind):
            h = h[0]
            Ngal = halos['Ngal'][h]
            vs = halos['vlos'][h][:Ngal]
            
            kern = gaussian_kde(vs)
            y = kern.resample(maxNgal - Ngal) # kern(pos) * maxv
            toobig = np.argwhere(np.abs(y) > maxv)
            
            while (len(toobig) > 0):
                y = np.delete(y, toobig)
                y = np.append(y, kern.resample(len(toobig)).flatten())
                toobig = np.argwhere(np.abs(y) > maxv)
            
            y = np.append(y.flatten(),vs)
            y = np.abs(y)
            
            featuresList.append(zip(y, np.ones(int(maxNgal))))
            
    feats = sdm.Features(featuresList, mass = massList, default_category = catname)
   


    if stand == True or stand == 'True' or stand == 'true':
        #standardize the features

        #if this is the first time through, choose a scaler:
        if scaler == None:
            print 'applying standardization;',
            feats, scaler = feats.standardize(ret_scaler=True)
            print 'scaler = ', scaler.mean_
        
        #otherwise, use the scaler that was passed:
        else:
            print 'standardizing features with scaler = ', scaler.mean_
            feats = feats.standardize(scaler = scaler)

    return feats, massList, IDList, scaler, numHalos
    




def predictUnlabeledData(trainhalos, testhalos):
    print ''
    print '=================  Predicting unlabeled halos ==================='

    # create train data from the train halos:
    traindata, trainmass, foldIDList, scaler, junk = \
                makeFeaturesList(trainhalos, None, 'train', featureType, stand, scaler = None)
    
    # and create test data from the test halos
    testdata, testmass, junk, junk, numHalos = \
         makeFeaturesList(testhalos, None, 'test', featureType, stand, scaler)

    preds = np.zeros((len(testmass), 3), 'f')

    # assign the training folds that 3-fold crossvalidation folds keep unique clusters separate
    # (NOTE: THIS WILL NOT WORK FOR ARBITRARILY SMALL NFOLDS, 
    # AND MIGHT DO SOMETHING REALLY DUMB FOR SOME NFOLDS
    # THIS WILL WORK BEST IF NFOLDS = 3*N+1, WHERE N IS SOME INTEGER.  
    # SO FOR 10 FOLDS, THIS WORKS NICELY, BUT PROCEED WITH CAUTION OTHERWISE
    print 'attempting to assign crossvalidation folds:'
    halo_folds = KFold(n=len(set(foldIDList)), n_folds=3, shuffle=True)
    model._tune_folds = [[np.vectorize(x.__contains__)(foldIDList) for x in traintest] \
                             for traintest in halo_folds]

    #this is a debugging relic that may prove useful later, so I'm leaving it in...
    #print 'checking isfinite:'
    #for item in [trainmass, testmass, traindata.mass]:
    #    print np.all(np.isfinite(item)),  np.min(item), np.max(item),len(set(item)), len(item)
    #    print ''

    
    #crossvalidate, put it in the 0th column of an array
    print 'crossvalidating'
    preds[:, 0] = \
            model.transduct(traindata, traindata.mass, testdata, save_fit=True)

    # put the true mass into the 1st column
    preds[: , 1] = testmass

    # print out some tuning parameters:
    print 'tuning parameters:', 
    print model._tuned_params()
    
    return preds

def predictLabeledData(n, halos):
    print ''
    print '=================  Fold', n, '==================='

    # create train data with all but one of the folds:
    trainFoldList = range(Nfolds)
    trainFoldList.remove(n)
    traindata, trainmass, foldIDList, scaler, junk = \
                makeFeaturesList(halos, trainFoldList, 'train', featureType, stand, \
                                     scaler = None)
    
    # and create test data with the fold of interest:
    testdata, testmass, junk, junk, numHalos = \
         makeFeaturesList(halos, [n], 'test', featureType, stand, scaler)

    #make an array to hold output:
    #0 = predicted mass
    #1 = true mass
    #2 = fold
    preds = np.zeros((len(testmass), 3), 'f')

    
    # assign the training folds (NOTE: THIS WILL NOT WORK FOR ARBITRARILY SMALL NFOLDS, 
    # AND MIGHT DO SOMETHING REALLY DUMB FOR SOME NFOLDS VALUES
    # THIS WILL WORK BEST IF NFOLDS = 3*N+1, WHERE N IS SOME INTEGER.  
    # SO FOR 10 FOLDS, THIS WORKS NICELY, BUT PROCEED WITH CAUTION OTHERWISE
    print 'attempting to assign crossvalidation folds:'
    halo_folds = KFold(n=len(set(foldIDList)), n_folds=3, shuffle=True)
    model._tune_folds = [[np.vectorize(x.__contains__)(foldIDList) for x in traintest] \
                             for traintest in halo_folds]


    #a debugging relic:
    #print 'checking isfinite:'
    #for item in [trainmass, testmass, traindata.mass]:
    #    print np.all(np.isfinite(item)),  np.min(item), np.max(item)
    #    print len(set(item)), len(item)
    #    print ''

    print 'traindata:', traindata
    print 'testdata:', testdata
    print 'model:', model
    #crossvalidate:
    print 'crossvalidating'
    preds[:, 0] = \
            model.transduct(traindata, traindata.mass, testdata, save_fit=True)

    # put the true mass into a file:
    preds[: , 1] = testmass
    # as well as the fold
    preds[: , 2] = n

    # print out some tuning parameters:
    print 'tuning parameters:', 
    print model._tuned_params()
    
    return preds


def run():
    
    # set up a timer
    t0 = time.clock()
    start_t = time.time()

    
    #count multiple runs
    log = listdir(logFolder)
    num = int(np.sum([(i[:(5+len(scenario))] == ('log_' + scenario + '_')) for i in log]))
    
    print("log folder at: " + logFolder + 'log_' + scenario+'_' + str(num) + '.txt')
    
    #ouptut progress to a file
    sys.stdout = open(logFolder + 'log_' + scenario+'_' + str(num) + '.txt', 'w', 1)

    print("~~~ start: %s ~~~" % str(start_t))

    #Messrs. Moony, Wormtail, Padfoot, and Prongs
    #Purveyors of Aids to Magical Mischief-Makers
    #are proud to present SDM predictions
    print 'I solemnly swear that I am up to no good.'

    #random seed:
    np.random.seed(seed)

    #get SDM ready to roll:
    global model
    model = setSDMModel(sdrModel, n_proc, divfunc, K)

    #load the halos:
    halos = np.load(trainFile)
    
    #subsample:
    if subsample != 1:
        halos = np.random.choice(halos, int(subsample*len(halos)), replace=False)

    #predict unlabeled file:
    if unlabeledFile != None:
        print 'predicting unlabeled data!'
        unlabeledClusters = np.load(unlabeledFile)
        preds = predictUnlabeledData(halos, unlabeledClusters)
        np.save(outFolder+'HeCSPredictions_'+redshift+'_' + str(num) + '.npy', preds)
        print 'unlabeled preds:'
        print preds

    #predict labeled data for each of the folds:
    preds = np.zeros((0,3), 'f')
    for n in range(Nfolds):
        preds = np.append(preds, predictLabeledData(n, halos))
        np.save(outFolder+scenario+'_' + str(num) + '_preds.npy', preds)

    print("~~~ %s hours ~~~" % ((time.time() - start_t) / (60*60) ))
    print 'Mischief managed.'
        
    return 0

"""
Excess code

    global redshift, trainFile, scenario, featureType, Nfolds, K, sdrModel, n_proc, seed, stand, outFolder, logFolder, unlabeledFile
    
    redshift = config['redshift']
    trainFile = config['trainFile']
    scenario = config['scenario']
    featureType = config['featureType']
    Nfolds = config['Nfolds']
    K = config['K']
    sdrModel = config['sdrModel']
    n_proc = config['n_proc']
    seed = config['seed']
    stand = config['stand']
    outFolder = config['outFolder']
    logFolder = config['logFolder']
    unlabeledFile = config['unlabeledFile']
    subsample = config['subsample']

"""
