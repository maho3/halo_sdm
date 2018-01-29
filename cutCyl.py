#!/usr/bin/env python2.7

#Usage: read in a catalog and cut cylinders around each cluster


import numpy as np
from array import array
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bisect import bisect_left
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.spatial import kdtree
import sys
sys.path.append('/home/cmhicks/tools/')
from cosmologyTools import *
#sys.setrecursionlimit(1000000)
#import sdm
#from massEstimators import *
#from plottingTools import *
#import argparse
import time
pi = math.pi

###################### FLAGS: ################################################################

debug=False            #do you want to run a quick debug cube?  Leave this set to False
outputtoFile = True    #do you want to output cylinders to a file?
makeNewCylCat = True   #do you want to make a new cylinder catalog?  Set to False after you've run one to save a TON of time
makeNewPadFile = True  #do you want to make a new padded box file?  Set to False after you've run one to save about an hour
makePureCat = True     #do you want to make a pure catalog, too?
simz = 0.117           #what redshift is the sim box?  


###################### PARAMETERS: ###########################################################

#the name of the simulation being used.  This sim name sets a number of parameters, defined later
simulation = 'MDPL2Rockstar'  #'nu2GC', 'Multidark', 'MDPL2Rockstar'

#what kind of clipping?  'sigclip' = iterative velocity paring according to some 
clip = 'sigclip'

#number of folds to assign.  Stick with 3n+1, where n is an integer.
Nfold = 10

#
Ntrain = 10

#how many objects should be in the flat mass function?  Note that this parameter is a bit of "wishful thinking" - you may end up with fewer than this number, with the mass function tapering off at the high mass end, if the number of rotations does not allow:
flatMF = 15000

#length of a small debugging box; not used frequently
Lcut = 200 #Mpc/h ... the cut used for the small debugging box

#number of subboxes; not used frequently
Nsubbox = 8**3

#minimum log(mass) to be a cluster candidate
logClusterMinMass = np.log10(1e14)  #minimum mass to be a "cluster candidate"

#minimum richness to be included in the catalog
richnessMin = 10 

#how large will the cylinder be?  I have a few pre-defined cylinders listed below:
cutparam = 'large'

#if objects have |v|>sigclip * sigma_v, they will be iteratively pared. To build an entire cylinder, just use a very large number here.
sigclip = 200.

if cutparam == 'small':
    aperture = 1.1     #the radial aperture in comoving Mpc/h
    vcut = 1570.       #maximum velocity is vmean +/- vcut in km/s
    ints15 = 569       #best fit sigma_15, for plotting purposes only
    intalpha = 0.209   #best fit alpha, for plotting purposes only
elif cutparam == 'medium':
    aperture = 1.6 #Mpc/h
    vcut = 2500. #km/s
    ints15 = 895
    intalpha = 0.384
elif cutparam == 'large':
    aperture = 2.3 #Mpc/h
    vcut = 3785. #km/s
    ints15 = 900.
    intalpha = 0.400
elif cutparam == 'custom':
    aperture = 1.6
    vcut = 2500.
    ints15 = None
    intalpha = None
#from ML with interlopers paper, we used the following values:
#             aperture (Mpc/h)         velocity cut (km/s)        sigma clip
#small:       1.1                      1570                       2 
#medium:      1.6                      2500                       2          ** used in paper body
#large:       2.3                      3785                       2

seed = 8675309 #Jenny.

np.random.seed(seed)


#the simulation sets parameters and file names:
if simulation == 'MDPL2Rockstar':
    maxNumGal = 1500       #maximum number of galaxies per cluster
    boxL = 1000            #box size in Mpc/h
    cosmoparams = {'OmegaM':0.307115,'OmegaL':0.692885 ,'Omegak':0.0, 'hubble':0.6777}
    G = 6.67408e-11 * (1/1000.)**2 * (1./3.086e22) * (1.9889200011445836e30/1.) 
    # G in units of km2 Mpc Msolar-1 s-2
    rhocrit = 3*100**2/(8*pi*G) #units = Msolar h2 Mpc-3
    w_DE = -1.             #simulation parameter
    rres = 0.05            #simulation resolution in Mpc/h, the softening length of the sim to define proper binning
    if debug == False:
        print 'running a full box'
        makeSmallFiles = False  #note: I haven't written small file capabilities for MD

        zstring = '{:.3f}'.format(simz)
        label = 'MDPL2Rockstar_z='+zstring
        txtout = label+'.txt'

        inFile = '/physics2/cmhicks/Multidark/MDPL2_lightcone/Rockstar_z='+zstring+\
                 '_Macc=1e11.csv'
    
        paddedFile = '/physics2/cmhicks/Multidark/MDPL2_lightcone/z='+zstring+'/1e11_padded.npy'
        pureCatFile = '/physics2/cmhicks/Multidark/MDPL2_lightcone/z='+zstring+'/pureclusters.npy'
        clusterFile = '/physics2/cmhicks/Multidark/MDPL2_lightcone/z='+zstring+'/clusters.npy' 
        if clip == 'sigclip':
            paredClusterFile = '/physics2/cmhicks/Multidark/MDPL2_lightcone/z='+zstring+\
                '/clusters_sigclip.npy'
        elif clip == 'caustic':
            paredClusterFile = '/physics2/cmhicks/Multidark/MDPL2_lightcone/z='+zstring+\
                '/clusters_caustic.npy'
        richnessFile = '/physics2/cmhicks/Multidark/MDPL2_lightcone/z='+zstring+\
            '/clusters_richness.npy'
        Nrot = 10           #number of rotations
        rotations = np.arange(Nrot)

    else:
        print 'error:  cannot run small boxes for MDPL2'
#############################################################################
G = 4.3016e-9 #Mpc/h km^2 (Msolar/h)^(-1) s^(-2)
pi = math.pi


def makeEmptyGalaxyFile(numGal):
    return np.zeros(numGal, dtype=[('uniqueID','uint32'), ('uniquehostID', 'uint32'),\
                                       ('morph','i'),('redshift', 'f'), ('redshift_sim', 'f'),\
                                       ('stellarmass','f'), ('gasmass', 'f'),\
                                       ('primarymass', 'f'), ('subhalomass','f'),\
                                       ('magnitude', 'f', 3), ('boxposition', 'f', 3),\
                                       ('velocity',  'f', 3), ('DMradius', 'f'),\
                                       ('usualsuspect', 'i'), ('bcg', 'bool'),\
                                       ('vlos', 'f'), ('Nsub_true', 'i'), ('Nrot', 'i'),\
                                       ('theta', 'f'), ('phi', 'f'), ('pix', 'i'),\
                                       ('M500', 'f'), ('Rs', 'f'), ('concentration', 'f'), \
                                       ('Macc', 'f')])

def makeDebugFile():

    # make a debug file that is short and easy to manage:
    dat = np.loadtxt(inFile)
    print len(dat)
    which = (dat[:,3] <=1) #just grabs a few subbox/trees from the original data.
    #another option, save dat[0:1253], which is a cut between cluster members, so it's easy
    np.savetxt(debuginFile, dat[which])
    print 'saved debug file'
    return None

def makeSmallBox():
    dat = np.loadtxt(inFile)
    which = (dat[:,12]<Lcut)&(dat[:,13]<Lcut)&(dat[:,14]<Lcut)
    print sum(which), 'of ', len(dat), ' galaxies are in the small box'
    np.savetxt(smallboxinFile, dat[which])
    return None
    

def readMDPL2RockstarGalaxyFile(): 
    # reads in my MDPL2 Rockstar file from my query.  
    #Below is a list of column names in the original query:
    #0 "row_id","rockstarId","upId","M200b","Rvir",
    #5 "x","y","z",
    #8 "vx","vy","vz",
    #11 "M500c","Rs","Macc","Vacc","Mvir","M200c"
    


    #itemList = ['ID', 'hostID', 'morph','tree', 'stmass', 'gasmass', 'primarymass', 'submass', 'q2', 'mag1', 'mag2', 'mag3', 'x', 'y', 'z', 'vx', 'vy', 'vz'] 

    #read in the file:
    dat = np.loadtxt(inFile, delimiter=',', skiprows=1)
    
    numGal = len(dat)
    print 'number of objects', numGal

    galaxies = makeEmptyGalaxyFile(numGal)
    
    #recently added:
    galaxies['M500'] = dat[:,11]
    galaxies['Rs'] = dat[:,12]/1000.
    galaxies['Macc'] = dat[:,13]
    

    
    galaxies['subhalomass'] = dat[:,13]  #we're going to start using mass at accretion!
    galaxies['DMradius'] = dat[:,4]/1000.  #MD put these in weird units; divide by 1000 to correct
    galaxies['concentration'] = galaxies['Rs']/galaxies['DMradius']
    #print 'Rockstar radius check for rhocrit:', dat[0:10,4], galaxies['DMradius'][0:10]
    for i in range(3):
        galaxies['boxposition'][:,i] = dat[:,5+i]
        galaxies['velocity'][:,i] = dat[:,8+i]
   
    galaxies['uniqueID'] = dat[:,1]
    #print 'ID check:', galaxies['uniqueID'][0:10]
    #must subtract or redefine to be a long int
    which = (dat[:,2]==-1)
    galaxies['uniquehostID'] = dat[:,2]
    #upID = rockstarId of most massive host halo, only different from pid, 
    #if halo lies within two or more larger halos

    #replace -1's with self's ID:
    print 'Of ', len(which), 'galaxies, ', sum(which), 'are hosts.'
    galaxies['uniquehostID'][which] = galaxies['uniqueID'][which]
    
    return galaxies




def calculatePaddingSize(v, z=0):
    #calculates the necessary padding for this particular sim, given vcut
    #returns padding in Mpc/h
    t0 = time.clock()
    #find the largest velocity in the sim:
    vmax = np.sqrt(np.max(v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2)) 
    pad = (vcut + vmax)/(100.)
    print 'Maximum peculiar velocity is ', '%.0f'%round(vmax,0), 'km/s,'
    print 'so a minimum pad of ', '%.0f'%round(pad+1,0), ' Mpc/h is needed.'
    return 1.05*pad

def pureCat(richness, galaxies):
    #creates and saves a pure catalog
    N = len(richness)
    pure = makeEmptyClusterFile(3*N, maxNumGal)

    print 'making a pure catalog.'
    
    for v in range(3):
        for j, h in enumerate(richness['ind']):
            los = [0.,0.,0.]
            los[v] = 1.
            perp = [(v+1)%3, (v+2)%3]
            Ngal = richness['richness'][j]

            i = N*v+j
            
            sInd = richness['membInd'][j][0:Ngal]

            if galaxies['usualsuspect'][h] == 0:
                print j, h, 'incorrect index'

            #note: I didn't fill in all the things in the array, just the ones I needed
            pure['Mtot'][i] = galaxies['subhalomass'][h]
            pure['M500'][i] = galaxies['M500'][h]
            pure['Rs'][i] = galaxies['Rs'][h]
            pure['hostid'][i] = galaxies['uniqueID'][h]
            pure['id'][i][0:Ngal] = richness['membID'][j][0:Ngal]
            pure['Mscale'][i][0:Ngal] = galaxies['subhalomass'][h]*np.ones(Ngal)
            pure['los'][i] = los
            pure['rotation'][i] = v
            pure['Ngal'][i] = Ngal
            pure['Msub'][i][0:Ngal] = galaxies['subhalomass'][sInd]
            pure['Macc'][i][0:Ngal] = galaxies['Macc'][sInd]
            pure['Rsub'][i][0:Ngal] = galaxies['DMradius'][sInd]
            pure['boxxyz'][i] = galaxies['boxposition'][h]
            pure['xyzsub'][i][0:Ngal] = galaxies['boxposition'][sInd]
            pure['vlos'][i][0:Ngal] = galaxies['velocity'][sInd, v]
            pure['sigmav'][i] = np.std(galaxies['velocity'][sInd, v])
            pure['R200'][i] = galaxies['DMradius'][h]
            #print 'debug run:', perp[0], perp[1]
            #print galaxies['boxposition'][sInd]
            #print galaxies['boxposition'][sInd,1]
            #print galaxies['boxposition'][sInd,perp[0]]
            pure['Rproj'][i][0:Ngal] = np.sqrt(\
            (galaxies['boxposition'][sInd, perp[0]]-galaxies['boxposition'][h, perp[0]])**2 + \
            (galaxies['boxposition'][sInd, perp[1]]-galaxies['boxposition'][h, perp[1]])**2)
            #print 'rproj check:',pure['Rproj'][i]
    # apply a power law fit (note that this is different from 1st & 2nd paper in that
    # 1) we're only using 3 LOS and
    # 2) no binning.
    print 'R200 check:', 
    print pure['R200'][0:10]
    print pure['Mtot'][0:10]
    density = pure['Mtot']/(pure['R200'])**3
    print 'density:', np.min(density), np.max(density), np.mean(density), np.median(density)
    m, b = checkMsig(pure, np.log10(3e14), 20, 'pure')
    pure = updatePLpreds(pure, m, b, pureCatFile)
            
    which = (pure['Ngal']>=richnessMin)

    pure = testTrainAndFold(pure[which])

    np.save(pureCatFile, pure)

def calculateTrueRichness(unsortedgalaxies, method = 'radius'):
    #returns an array of ids and true richness, using either radius (preferred) or IDs (a little too trusting of Multidark for my taste!)
   

    #loop & look...
    pstart = 0
    print 'matching hosts...',

    if method == 'ID':
        
        #unless we want to spend a couple of hours here, sort host IDs:
        ind =  np.argsort(unsortedgalaxies['uniquehostID'])
        galaxies = unsortedgalaxies[ind]
        Ngalaxies = len(galaxies)

        #find the hosts:
        ishost = (galaxies['usualsuspect'] == 1)
        numHost = sum(ishost)
    
    
        # set up an array to hold ID and true richness
        richness = np.zeros(numHost, dtype = [('uniquehostID', 'i'), ('richness', 'i'), \
                                          ('membID', 'i', maxNumGal), ('ind', 'i'), \
                                          ('membInd', 'i', maxNumGal), ('Mtot', 'f')])


        hostIDset = np.sort(np.asarray(list(set(galaxies['uniquehostID'][ishost]))))
        richness['uniquehostID'] = hostIDset
        #search by the ID of the host:
        for i, ID in enumerate(hostIDset):
            hostFound = False
            for p in pstart + np.arange(Ngalaxies-pstart):
                # search forward to find the first object with this hostID:
                if hostFound == False and galaxies['uniquehostID'][p] == ID:
                    hostFound = True
                    N = 0
                    qstart = p
                if hostFound == True:
                    if galaxies['uniquehostID'][p] == ID:
                        N += 1
                    else:
                        hostFound = False
                        richness['richness'][i] = N
                        richness['membID'][i][0:N] = galaxies['uniqueID'][qstart:p]
                        pstart = p
                        break

    elif method == 'radius':

        galaxies = unsortedgalaxies

        #find the hosts:
        ishost = (galaxies['usualsuspect'] == 1)
        numHost = sum(ishost)
    
    
        # set up an array to hold ID and true richness
        richness = np.zeros(numHost, dtype = [('uniquehostID', 'i'), ('richness', 'i'), \
                                          ('membID', 'i', maxNumGal), ('ind', 'i'), \
                                          ('membInd', 'i', maxNumGal), ('Mtot', 'f'),\
                                          ('mfrac', 'f')])

        #print 'radius check:', galaxies['DMradius'][0:10]
        #print 'searching for cluster membership through radius...'
        tree = kdtree.KDTree(galaxies['boxposition'])
        #print 'tree finished.'
        hind = np.where(galaxies['usualsuspect']==1)[0]
        #print 'hind check:', hind[0:10]
        for i, h in enumerate(hind):
            sInd = np.asarray(tree.query_ball_point(\
                    galaxies['boxposition'][h], galaxies['DMradius'][h]))
            N = len(sInd)
            richness['uniquehostID'][i] = galaxies['uniqueID'][h]
            richness['richness'][i] = N
            richness['membID'][i][0:N] = galaxies['uniqueID'][sInd]
            richness['ind'][i] = h
            richness['membInd'][i][0:N] = sInd
            richness['Mtot'][i] = galaxies['subhalomass'][h]
            richness['mfrac'][i] = np.sum(galaxies['subhalomass'][sInd])/galaxies['subhalomass'][h]
                    
    
    np.save(richnessFile, richness)
    return richness



def checkMsig(cluster, logMcut, richcut, label):
    #check the M(sigma) relation:



    which = (np.log10(cluster['Mtot']) > logMcut) & (cluster['Ngal'] > richcut)

    logM = np.log10(cluster['Mtot'][which])
    logsig = np.log10(cluster['sigmav'][which])

    plt.scatter(np.log10(cluster['Mtot']), np.log10(cluster['sigmav']), s=2, lw=0, alpha=0.05, c='k', label = '6 rotations')
    plt.xlabel(r'$\log(M_\mathrm{true})$')
    plt.ylabel(r'$\log(\sigma_v)$')
    plt.xlim(14, 15.5)
    plt.ylim(1.8, 3.5)

    plt.plot([logMcut, logMcut], [1.8, 3.5], c='k', ls='-.')

    m, b = np.polyfit(logM, logsig, 1)
    print 'fit parameters: ', m, b

    x = np.linspace(14, 15.5, 20)
    y = m*x + b
    plt.plot(x, y, ls='-', c='r')

    sigfit = m*logM + b
    scatter = np.std(logsig-sigfit)

    for i in [-1., 1.]:
        plt.plot(x, y+1*i*scatter, ls='--', c='r')
        plt.plot(x, y+2*i*scatter, ls=':', c='r')
    
    s15 = 10**b * (10**15)**m
    alpha = m
    eqs15 = r'$\sigma_{15} = '+'%.0f' % round(s15,0)+'$'
    eqalp = r'$\alpha = '+'%.2f' % round(alpha,3)+'$'
    plt.annotate(eqs15, xy=(0.1, 0.9),  xycoords = 'axes fraction')
    plt.annotate(eqalp, xy=(0.1, 0.85), xycoords = 'axes fraction')
            
    plt.savefig('output/Msigcheck_'+label+'.png')
    plt.clf()


    logMpred = (np.log10(cluster['sigmav'])-b)/m
    plt.scatter(np.log10(cluster['Mtot']), logMpred, s=2, lw=0, \
                    alpha=0.05, c='k', label = '6 rotations')
    plt.xlabel(r'$\log(M_\mathrm{true})$')
    plt.xlabel(r'$\log(M_\mathrm{pred})$')
    plt.xlim(14, 15.5)
    plt.ylim(14, 15.5)

    plt.plot(x, x, ls='-', c='r')
    plt.savefig('output/Mpred_'+label+'.png')
    plt.clf()
    


    return m, b

def makeEmptyClusterFile(Nhalo, Ngal):
    #Mtot = true cluster mass
    #id = cluster ID
    #los = line of sight used to view this cluster [x, y, z]
    #Msub = subhalo mass
    #vlos in the los z, such that mean(vlos) = 0
    #sigmav = std(vlos)
    #Rproj = projected dist = sqrt(x^2 + y^2)
    #lum = galaxy luminosity
    #color = galaxy color (list bands here)
    #memb = boolean array telling whether it's a true cluster member or not
    #clusterxyz = positional center of cluster in the box
    #subbox = boolean array telling whether a cluster is in a subbox or not
    #intest = boolean telling if it should be in the test sample 
    #intrain = boolean telling if it should be in the train sample

    return np.zeros(Nhalo, dtype=[('Mtot', 'f'), ('hostid', 'uint32'), ('id', 'uint32', Ngal),\
                           ('los', 'f', 3), ('R200', 'f'), ('rotation', 'i'),\
                           ('fold', 'i'), ('Ngal', 'i'), ('Ngaltrue', 'i'),\
                           ('Msub', 'f', Ngal), ('Rsub', 'f', Ngal), ('vlos', 'f', Ngal),\
                           ('sigmav', 'f'), ('xyzsub', 'f', (Ngal,3)),\
                           ('Rproj', 'f', Ngal), ('xyproj', 'f', (Ngal,2)),\
                           ('lum', 'f', (Ngal,3)), ('color', 'f', Ngal),\
                           ('purity', 'f'), ('completeness', 'f'), ('pix', 'i'),\
                           ('boxxyz', 'f', 3), ('name', 'str', 20), ('Mxray', 'f'),\
                           ('subbox', bool, Nsubbox), ('truememb', bool, Ngal),\
                           ('intest', 'i'), ('intrain', 'i'), ('redshift', 'f'),\
                           ('PL1pred', 'f'), ('vhubble', 'f'), ('Mscale', 'f', Ngal),\
                           ('YSZ', 'f'), ('M500', 'f'), ('Rs', 'f'), ('concentration', 'f'),\
                           ('Macc', 'f', Ngal), ('MSZ', 'f'), ('sig_KR', 'f')])


def padGalaxyFile(galaxies, padsize, paddedFile):
    i = 0
    paddedgalaxies = np.zeros(0, 'f')
    for ix in [0, -1, 1]:
        for iy in [0, -1, 1]:
            for iz in [0, -1, 1]:
                

                marker = abs(ix)+abs(iy)+abs(iz)

                if marker == 0:
                    print 'box center'
                    paddedgalaxies = np.copy(galaxies)
                else:
                    #make a copy of the box, and apply a shift:
                    temp = np.copy(galaxies)
                    temp['boxposition'][:,0] += ix*boxL
                    temp['boxposition'][:,1] += iy*boxL
                    temp['boxposition'][:,2] += iz*boxL

                    #find a mask:
                    which = (temp['boxposition'][:,0] > (-padsize))&\
                        (temp['boxposition'][:,0] < (boxL+padsize))&\
                        (temp['boxposition'][:,1] > (-padsize))&\
                        (temp['boxposition'][:,1] < (boxL+padsize))&\
                        (temp['boxposition'][:,2] > (-padsize))&\
                        (temp['boxposition'][:,2] < (boxL+padsize))


                    paddedgalaxies = np.append(paddedgalaxies, temp[which])

                #some debugging outputs:
                print 'padding ', i, ':    ', ix, iy, iz
                i += 1
                

    print 'new code - min and max positions:'
    for n in range(3):
        print n, np.min(paddedgalaxies['boxposition'][:,n]), 
        print np.max(paddedgalaxies['boxposition'][:,n]), 
    print 'Original file has ', len(galaxies), 'galaxies.'
    print 'Box replicated 3x3x3 times has', len(paddedgalaxies), 'galaxies.', 
    #print 'Saving padded box with', sum(which), 'galaxies.'


    np.save(paddedFile, paddedgalaxies)
    return paddedFile

def determinebasis(n):
    #for n = 0, 1, or 2, will return a LOS along the box
    #for n >= 3, will return an r randomly oriented on the unit sphere
    if n <= 2:
        r = [0.,0.,0.]
        r[n] = 1.
        rperp = [[r[1], r[2], r[0]], [r[2], r[0], r[1]]]
    else:
        # randomly choose a location on the unit hemisphere
        # r=[x, y, z]
        # with -1<x<1, -1<y<1, and 0<z<1
        # and x^2+y^2+z^2 = 1
        # (I think the upper limits are actually <=, but double-check if this becomes important)
        phi = 2*pi*np.random.random()
        theta = np.arccos(np.random.random())
        r = normVector([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

        # Find 2 other vectors that are orthonormal to the first:
        rperp = planeVectors(r)
    basis = [r, rperp[0], rperp[1]]

    return basis

def normVector(v):
    #returns a positive, normalized vector
    v = v / (np.sqrt(np.dot(v,v)))
    return v


def planeVectors(v):
    #returns two vectors that are 
    #orthogonal to the vector v, orthogonal to each other, and normalized

    if v[0]==1. or v[1]==1. or v[2]==1.:
        #v lies along a box direction
        r1 = normVector([v[1], v[2], v[0]])
    else:
        #fewer than 2 of the components of v are 0
        r1 = normVector([v[1], -v[0], 0])
        
    r2 = normVector(-np.cross(v, r1))
    return [r1, r2]


def findBCG(galaxies, method = 'host'):
    if method == 'host':
        bcgbool = (galaxies['uniqueID'] == galaxies['uniquehostID'])
    return bcgbool

def findUsualSuspects(galaxies, method='minmass'):
    if method == 'minmass':
        suspects = (galaxies['boxposition'][:,0]>=0) & (galaxies['boxposition'][:,0]<boxL) & \
                (galaxies['boxposition'][:,1]>=0) & (galaxies['boxposition'][:,1]<boxL) & \
                (galaxies['boxposition'][:,2]>=0) & (galaxies['boxposition'][:,2]<boxL) & \
                (galaxies['subhalomass']>10**logClusterMinMass) & (galaxies['bcg'] == True)
    elif method == 'debug':
        suspects = (galaxies['boxposition'][:,0]>=0) & (galaxies['boxposition'][:,0]<boxL) & \
                (galaxies['boxposition'][:,1]>=0) & (galaxies['boxposition'][:,1]<boxL) & \
                (galaxies['boxposition'][:,2]>=0) & (galaxies['boxposition'][:,2]<boxL) & \
                (galaxies['subhalomass']>10**14.8) & (galaxies['bcg'] == True)
    #using subhalomass cuts is a little misleading... galaxies['bcg']== True ensures that I'm looking at a host, but I really ought to put in a step that matches up subs to their hosts asap.
    #to do:  match subs to the hosts before this step, with a nice sort.
            
    print sum(suspects), ' usual suspects identified of', len(suspects), 'galaxies.'
    return suspects

def rotateObserver(boxoriented, basis):
    #rotates observer, returns (x', y') position, z' distance, and vz' LOS velocity
    #where the primes denote that we're not working in the box direction anymore, but rather
    #in some arbitrary basis where x', y', and z' are orthonormal
    
    #make an empty file and copy over the relevant info that is unchanged under rotation
    numGal = len(boxoriented)
    rotatedbox = makeEmptyGalaxyFile(numGal)

    for item in ['uniqueID', 'uniquehostID', 'morph', 'redshift', 'redshift_sim',\
                     'stellarmass', 'gasmass', 'primarymass', 'subhalomass', 'magnitude', \
                     'DMradius', 'usualsuspect', 'bcg', 'Nrot', 'M500', 'Rs', 'concentration',\
                     'Macc']:
        rotatedbox[item] = boxoriented[item]




    #peculiar velocity along the los (p=0) and the perpendicular directions (p=1, p=2)
    for p in range(3):
        for n in range(3):
            rotatedbox['velocity'][:,p] += basis[p][n]*boxoriented['velocity'][:,n]

    #positions in the new basis:
    for p in range(3):
        for n in range(3):
            rotatedbox['boxposition'][:,p] += basis[p][n]*boxoriented['boxposition'][:,n]
    
    
    return rotatedbox

def cutCylinders(galaxies, unrotatedBox, aperture, rotation, basis, boxoriented):
    #this is really the workhorse of the code.  We'll take a rotated box and cut cylinders
    #around all of the "usual suspects"!

    print 'cutting cylinders around rotation ', rotation

    #default is to look down the x los, with y and z defining the plane of sky,
    #but write flexibly enough to change!
    los = 0
    perp = [(los+1)%3, (los+2)%3]

    #even though the grid is 3D, I just want to find nearest neighbors in 2D 
    #(i.e. y and z, defined by the perp vector)
    tstart = time.time()
   
    #position = np.asarray([galaxies['boxposition'][:,perp[0]], galaxies['boxposition'][:,perp[1]]])
    position = np.asarray([[galaxies['boxposition'][i,perp[0]], \
                                galaxies['boxposition'][i,perp[1]]] \
                               for i in range(len(galaxies))])
    #for i in range(10):
    #    print i, position[i]
    #for i in range(2):
    #    print np.min(position[:,i]), np.max(position[:,i])

    plt.hist2d(position[:,0], position[:,1], bins=100)
    plt.colorbar()
    plt.savefig('output/'+str(rotation)+'_hist2d.png')
    plt.clf()

    tree = kdtree.KDTree(position, leafsize=50)
    print 'it takes ', (time.time()-tstart)/60., 'minutes to make a tree.'

    print 'determining h indices'
    hInd = np.where(galaxies['usualsuspect']==1)[0]
    print 'hInd old:', len(hInd), hInd[0:10]
    if rotation < 3:
        print 'rot < 3'
        hInd = np.argwhere(galaxies['usualsuspect']==1).flatten()#reshape(-1, len(hInd))[0]
    else:
        print 'rot >= 3'
        hInd = np.argwhere((galaxies['usualsuspect']==1) &\
                               ( galaxies['Nrot'] >= rotation+1)).flatten()#reshape(-1, len(hInd))[0]
    print 'hInd new:', len(hInd), hInd[0:10]

    #how many clusters do we expect?  Make an empty array to hold it all!
    print 'making empty array'
    Nsuspects = len(hInd)
    cluster = makeEmptyClusterFile(Nsuspects, maxNumGal)
   
    #load the richness file:
    richness = np.load(richnessFile)

    maxN = 0
    vzero = observedVlos(0.0, z=simz, **cosmoparams) #hubble flow at sim redshift
    
    t0 = time.clock()
    print 'number of clusters: ', len(hInd)
    for i, h in enumerate(hInd):
        
        sInd = np.asarray(tree.query_ball_point(\
            [galaxies['boxposition'][h,perp[0]], galaxies['boxposition'][h,perp[1]]],\
            aperture))
        
        #vctr = observedVlos(galaxies['velocity'][h,los], z=simz)  #cluster center with pecular velocity
        

        # and look at hubble flow of all of the halos within the aperature:
        vsub = observedVlos(galaxies['velocity'][sInd,los], \
                          Dc = galaxies['boxposition'][sInd,los]-galaxies['boxposition'][h, los]+\
                          z_to_Dc([simz], **cosmoparams), **cosmoparams)
        #the second passed term above is confusing... 
        #Briefly, it looks at how far the galaxies in the cylinder relative to that of the 
        #halo of interest, and parks the halo of interest a set distance away.
        #That set distance is whatever comoving distance corresponds to the sim redshift.
        
        
        
        #not going to do this for right now... if it's reinstated, it needs to be checked 
        #and debugged.
        # let's keep the distance to each galaxy along the los, too.
        #subdist = galaxies['boxposition'][sInd, los]-\
        #          (galaxies['boxposition'][h, los]-dhalo)
        
        
        # now, I want to define a cylinder height, and pare my sample further 
        # to only the points that are within the cylinder:
        imask = abs(vsub-vzero)<vcut  #a boolean mask of the subhalo index list
        incyl = sInd[imask]  # the galaxies with these indices are IN the cylinder, 
                         #so this is, in some sense, the "final answer"!
        
        #how many galaxies, including BCG, are in the cylinder?
        Ngal = len(incyl) 

        #to do:  true membership / completeness is computationally expensive.  
        #Try a sort to just do this once
        #how many galaxies, including BCG, are truly cluster members?
        #Ntrue = sum(galaxies['uniquehostID'] == galaxies['uniqueID'][h])
        #how many galaxies, including BCG, are actually cluster members
        #and are also in the cylinder?
        truememb = (galaxies['uniquehostID'][incyl] == galaxies['uniqueID'][h])
        
        Nboth = sum(truememb)
        #how many galaxies are interlopers?
        Nint = sum(galaxies['uniquehostID'][incyl] != galaxies['uniqueID'][h])
        #print (time.clock()-tloop)/60.
        if h < 5:
            print h, Nboth, Nint, Ngal
            print truememb
            print ''

        #fill in the cluster info:
        #print 'array...',
        cluster['Ngal'][i] = Ngal
        #note that we know that the galaxy at 'h' is a host, but if we ever
        #search for cluster candidates, this would need to be changed:
        cluster['Mtot'][i] = galaxies['subhalomass'][h] 
        cluster['M500'][i] = galaxies['M500'][h]
        cluster['hostid'][i] = galaxies['uniqueID'][h] #the host ID
        cluster['id'][i,0:Ngal] = galaxies['uniqueID'][incyl]
        cluster['Mscale'][i,0:Ngal] = galaxies['primarymass'][incyl]
        #xyz position from **original box** to eventually determine subboxes
        cluster['boxxyz'][i] = unrotatedBox['boxposition'][h]
        cluster['xyzsub'][i,0:Ngal] = unrotatedBox['boxposition'][incyl]
        
        cluster['los'][i] = basis[los]
        cluster['rotation'][i] = rotation
        cluster['vhubble'] = vzero

        #to do: cluster['fold'] = None
        cluster['Msub'][i,0:Ngal] = galaxies['subhalomass'][incyl]
        cluster['Macc'][i,0:Ngal] = galaxies['Macc'][incyl]
        cluster['Rsub'][i,0:Ngal] = galaxies['DMradius'][incyl]
        cluster['vlos'][i,0:Ngal] = galaxies['velocity'][incyl,los]
        cluster['sigmav'][i] = np.std(cluster['vlos'][i,0:Ngal])
        cluster['Rproj'][i,0:Ngal] = np.sqrt(\
            (galaxies['boxposition'][incyl,perp[0]]-galaxies['boxposition'][h,perp[0]])**2 + \
            (galaxies['boxposition'][incyl,perp[1]]-galaxies['boxposition'][h,perp[1]])**2)
        cluster['xyproj'][i,0:Ngal,0] = \
            galaxies['boxposition'][incyl,perp[0]]-galaxies['boxposition'][h,perp[0]]
        cluster['xyproj'][i,0:Ngal,1] = \
             galaxies['boxposition'][incyl,perp[1]]-galaxies['boxposition'][h,perp[1]]
        cluster['lum'][i,0:Ngal] = galaxies['magnitude'][incyl]
        #cluster['color'][i] = None
        #cluster['completeness'][i] = 100*float(Nboth)/float(Ntrue)
        cluster['truememb'][i][0:Ngal] = truememb
        cluster['purity'][i] = 100*sum(truememb)/float(len(truememb))
        cluster['Ngaltrue'][i] = richness['richness'][i]
        
        if rotation < 3:
            cluster['intest'][i] = 1
        else:
            cluster['intest'][i] = 0

        
    print 'max number of members found:', max(cluster['Ngal'])
        
    return cluster


def makeCylCat(t0):
    

    if debug == False:

        if makeSmallFiles == True:

            print 'making a debug file...',
            makeDebugFile() 
            print 'finished in ',(time.clock()-t0)/60., 'minutes.'
            print ''

            print 'making a small box file...',
            makeSmallBox() 
            print 'finished in ',(time.clock()-t0)/60., 'minutes.'
            print ''
 
    print 'reading and processing galaxy file...',
    if simulation == 'nu2gc':
        galaxies = readnu2GCGalaxyFile()
    elif simulation == 'Multidark':
        galaxies = readMDGalaxyFile()
    elif simulation == 'MDPL2Rockstar':
        galaxies = readMDPL2RockstarGalaxyFile()
    else: 
        print 'cannot process this simulation.'
    print 'finished in ', (time.clock()-t0)/60., 'minutes.'
    print ''
    
              
    tpad = time.clock()
    if makeNewPadFile == True:
        print 're-generating padded file...',
        padsize = calculatePaddingSize(galaxies['velocity'])
        padGalaxyFile(galaxies, padsize, paddedFile)
        print 'finished in ', (time.clock()-t0)/60., 'minutes.'
        print ''
    print 'loading padded file...', 
    paddedgal = np.load(paddedFile)
    print 'finished in ', (time.clock()-t0)/60., 'minutes.'
    print ''

    print 'finding the BCGs...',
    paddedgal['bcg'] = findBCG(paddedgal)
    which = paddedgal['bcg']== True
    print 'number of bcg:', sum(which)
    #print np.min(paddedgal['subhalomass'][which]), 
    #print np.max(paddedgal['subhalomass'][which])
    print 'finished in ', (time.clock()-t0)/60., 'minutes.'
    print ''

    print 'finding the usual suspects...',
    paddedgal['usualsuspect'] = findUsualSuspects(paddedgal)
    #paddedgal['Nrot'] = calculateNrot(paddedgal)
    print 'number of usual suspects:', sum(paddedgal['usualsuspect']==True)
    print 'finished in ', (time.clock()-t0)/60., 'minutes.'
    print ''

    np.save(paddedFile, paddedgal)
    
    print 'calculating true richness...',
    #caution: this bit takes ~10 mins for a very small payoff: the true number
    #         of members in each cluster.  Best to comment this out unless the info
    #         is actually necessary.
    galaxies['bcg'] = findBCG(galaxies)
    galaxies['usualsuspect'] = findUsualSuspects(galaxies)
    richness = calculateTrueRichness(galaxies)
    print 'finished in ', (time.clock()-t0)/60., 'minutes.'
    print ''

    if makePureCat == True:
        print 'making pure catalog...', 
        pureCat(richness, galaxies)
        print 'finished in ', (time.clock()-t0)/60., 'minutes.'
        print ''
    
    print 'making a cluster file...',
    clusters = makeEmptyClusterFile(0, maxNumGal)
    print 'finished in ', (time.clock()-t0)/60., 'minutes.'
    print ''

    

    for n in rotations:
        print 'rotation #', str(n), '...'
        basis = determinebasis(n)
        rotatedBox = rotateObserver(paddedgal, basis)
        clusters = np.append(clusters, cutCylinders(rotatedBox, paddedgal, aperture, n, basis, paddedgal))
        np.save(clusterFile, clusters)
        
        print 'finished in ', (time.clock()-t0)/60., 'minutes.'
        print ''

    
    return None

def checkCyl(clusters, logMcut, richcut, label):
    #checks completeness, M(richness), 

    for item in ['Mtot', 'M500', 'sigmav', 'Ngal', 'Ngaltrue']:
        print item, clusters[item][0:10]
    logM = np.log10(clusters['Mtot'])
    which = (logM > logMcut) & (clusters['Ngal']>richcut)

    #check the M(richness) relationship:
    print 'checking measured richness...'
    plt.scatter(logM[which], np.log10(clusters['Ngal'][which]), s=2, lw=0, alpha=0.3)
    plt.xlabel(r'$\log(M_\mathrm{true})$')
    plt.ylabel(r'$\log(\mathcal{R})$')
    plt.savefig('output/richness_'+label+'.png')
    plt.clf()

    #check the true M(richness) relationship:
    print 'checking true richness...'
    plt.scatter(logM[which], np.log10(clusters['Ngaltrue'][which]), s=2, lw=0, alpha=0.3)
    plt.xlabel(r'$\log(M_\mathrm{true})$')
    plt.ylabel(r'$\log(\mathcal{R})$')
    plt.savefig('output/richness_true_'+label+'.png')
    plt.clf()

    #check the M(purity) relationship:
    print 'checking purity...'
    plt.scatter(logM[which], clusters['purity'][which], s=2, lw=0, alpha=0.3)
    plt.xlabel(r'$\log(M_\mathrm{true})$')
    plt.ylabel('purity')
    plt.savefig('output/purity_'+label+'.png')
    plt.clf()

    #check the M(completeness) relationship:
    print 'checking completeness...'
    plt.scatter(logM[which], clusters['completeness'][which], s=2, lw=0, alpha=0.3)
    plt.xlabel(r'$\log(M_\mathrm{true})$')
    plt.ylabel('completeness')
    plt.savefig('output/completeness_'+label+'.png')
    plt.clf()
    
    return None





def cuts(cluster, richcut = None, masscut = None):
    if richcut != None:
        which = (cluster['Ngal']>= richcut)
        print sum(which), ' of ', len(which), 'clusters have the minimum richness.'
        cluster = cluster[which]
    if masscut != None:
        which = (cluster['Mtot']>= masscut)
        print sum(which), ' of ', len(which), 'clusters have the minimum mass.'
        cluster = cluster[which]
    return cluster

def sigPare(cluster, sigclip, richcut = richnessMin, vcut = vcut, Rcut = aperture, Rcut_min = 0., nmax = 30, paredClusterFile = paredClusterFile, howzero = 'mean', customzero = None, folds = True, Mcut = None, recenter = False, vcenter = 'mean'):

    #applies an interative clipping to the clusters in velocity space
    #options for recenter:  'CM', 'CM_iterative', False
    #options for vcenter:  'mean', 'median', False


    print '================== SIGPARE PARAMS =================='
    print 'Rcut (aperture):', Rcut
    print 'vcut:', vcut
    print 'Rcut_min:', Rcut_min
    print 'howzero', howzero
    print 'recenter', recenter
    print 'vcenter', vcenter
    print 
    print '===================================================='

    print 'sig paring with ', richcut, 'minimum members.'
    #make an empty file:
    Nhalo = len(cluster)
    Ngalmax = len(cluster['vlos'][0])

    paredCluster = makeEmptyClusterFile(Nhalo, Ngalmax)

    Nclusters = len(cluster)
    s = len(cluster['vlos'][0]) #assumes they're all the same size
    
    if Mcut is not None:
        print 'applying a mass cut!'

    if recenter == False:
        print 'using the given cluster center'
    elif recenter == 'CM':
        print 'recentering to CM'
    elif recenter == 'CM_iterative':
        print 'iteratively recentering to CM'
        Nit_max = 20 #max number of iterations
        Rconverge = 0.02 #radius that defines convergence
        vconverge = 50.
    else:
        print 'error:  recentering method not found!'
        print crash
            

    for h in range(Nclusters):
        Ngal = cluster['Ngal'][h]
        if h < 2:
            print Ngal, 
        Nold = Ngal
        vlos = cluster['vlos'][h][0:Ngal]

        #apply a recentering:
        if recenter == 'CM':
            if h == 0:
                print 'recentering using the center of mass.'
            #cut an initial cylinder
            if Mcut == None:
                mask = ((cluster['Rproj'][h][0:Ngal]<Rcut)&\
                        (np.abs(cluster['vlos'][h][0:Ngal])<vcut))
            else:
                mask = ((cluster['Rproj'][h][0:Ngal]<Rcut)&\
                        (np.abs(cluster['vlos'][h][0:Ngal])<vcut)&\
                        (cluster['Msub'][h][0:Ngal]>Mcut))
            #calculate the center of mass
            xoffset = np.sum(cluster['xyproj'][h,0:Ngal,0][mask])/float(sum(mask))
            yoffset = np.sum(cluster['xyproj'][h,0:Ngal,1][mask])/float(sum(mask))
        elif recenter == 'CM_iterative':
            if h == 0:
                print 'iteratively recentering using the center of mass.'
            # cut an initial cylinder
            if Mcut == None:
                mask = ((cluster['Rproj'][h][0:Ngal]<Rcut)&\
                        (np.abs(cluster['vlos'][h][0:Ngal])<vcut))
            else:
                mask = ((cluster['Rproj'][h][0:Ngal]<Rcut)&\
                        (np.abs(cluster['vlos'][h][0:Ngal])<vcut)&\
                        (cluster['Msub'][h][0:Ngal]>Mcut))
            #and loop through a number of iterations until it converges.    
            for n in range(Nit_max):
                # calculate the center of mass
                xoffset = np.sum(cluster['xyproj'][h,0:Ngal,0][mask])/float(sum(mask))
                yoffset = np.sum(cluster['xyproj'][h,0:Ngal,1][mask])/float(sum(mask))
                if vcenter == 'mean':
                    voffset = np.mean(cluster['vlos'][h][0:Ngal][mask])
                elif vcenter == 'median':
                    voffset = np.median(cluster['vlos'][h][0:Ngal][mask])
                else:
                    voffset = 0.0

                if (np.sqrt(xoffset**2+yoffset**2)<Rconverge) and (voffset < vconverge):
                    # the recentering has converged.  Call it a day
                    if h < 50:
                        print 'cluster', h, 'converged in ', n, 'iterations:', 
                        print np.sqrt(xoffset**2+yoffset**2), voffset
                    break
                else:
                    #re-calculate all radius values:
                    cluster['xyproj'][h,0:Ngal,0] -= xoffset
                    cluster['xyproj'][h,0:Ngal,1] -= yoffset
                    cluster['vlos'][h,0:Ngal] -= voffset
                    cluster['Rproj'][h,0:Ngal] = np.sqrt(cluster['xyproj'][h,0:Ngal,0]**2+\
                                             cluster['xyproj'][h,0:Ngal,1]**2)
                    #apply a new mask:
                    if Mcut == None:
                        mask = ((cluster['Rproj'][h][0:Ngal]<Rcut)&\
                                    (np.abs(cluster['vlos'][h][0:Ngal])<vcut))
                    else:
                        mask = ((cluster['Rproj'][h][0:Ngal]<Rcut)&\
                                    (np.abs(cluster['vlos'][h][0:Ngal])<vcut)&\
                                    (cluster['Msub'][h][0:Ngal]>Mcut))
                
            
        else:
            if h == 0:
                print 'using original center.'
            xoffset = 0.
            yoffset = 0.
            voffset = 0.

        #re-calculate all radius values:
        cluster['xyproj'][h,0:Ngal,0] -= xoffset
        cluster['xyproj'][h,0:Ngal,1] -= yoffset
        cluster['vlos'][h,0:Ngal] -= voffset
        #cluster['Rsub'][h] is the RADIUS OF THE DARK MATTER... THIS IS NOT AN ISSUE HERE!
        cluster['Rproj'][h,0:Ngal] = np.sqrt(cluster['xyproj'][h,0:Ngal,0]**2+\
                                             cluster['xyproj'][h,0:Ngal,1]**2)

        #cut the initial cylinder, including a minimum radius (i.e. ignore the center!)
        if Mcut == None:
            mask = ((cluster['Rproj'][h][0:Ngal]<Rcut)&(cluster['Rproj'][h][0:Ngal]>=Rcut_min)&\
                        (np.abs(cluster['vlos'][h][0:Ngal])<vcut))
        else:
            mask = ((cluster['Rproj'][h][0:Ngal]<Rcut)&(cluster['Rproj'][h][0:Ngal]>=Rcut_min)&\
                        (np.abs(cluster['vlos'][h][0:Ngal])<vcut)&\
                        (cluster['Msub'][h][0:Ngal]>Mcut))
            if h == 0:
                print 'you have entered the Mcut loop successfully!'
        
        
        for n in range(nmax):
            if howzero == 'mean':
                vzero = np.mean(vlos[mask])
            elif howzero == 'custom':
                vzero = customzero[h]

            
                
            vlos = vlos - vzero
            sig = np.std(vlos[mask])
            mask[np.abs(vlos)> (sigclip * sig)] = False

            
            if n == nmax:
                    print 'max iterations reached for halo', h, clusters['id'][h]
            
            if sum(mask)==Nold:
                #print 'end of the line:', h, n, Nold
                paredCluster['Ngal'][h] = Nold
                #print 'eol check 2:', paredCluster['Ngal'][h]
                #stuff that copies over directly:
                for item in ['Mtot', 'M500', 'Rs', 'MSZ', 'concentration', \
                                 'hostid', 'los', 'rotation', 'fold',\
                                 'boxxyz', 'subbox', 'intest', 'intrain', 'name',\
                                 'redshift', 'YSZ', 'pix',  'Ngaltrue']:
                    paredCluster[item][h] = cluster[item][h]
                #stuff that copies over for the galaxies that remain:
                for item in ['vlos', 'id', 'Rproj', 'xyproj', 'Msub', 'Macc', 'Rsub', 'xyzsub', 'lum', 'color', 'truememb']:
                    paredCluster[item][h][0:Nold] = cluster[item][h][0:Ngal][mask]
                #if h <2:
                #    print 'los check 2:', h, paredCluster['Ngal'][h]


                #stuff that needs to be recalculated:
                paredCluster['sigmav'][h] = np.std(cluster['vlos'][h][0:Ngal][mask])

                paredCluster['purity'][h] = \
                    100*sum(paredCluster['id'][h] == paredCluster['hostid'][h])/float(Nold)
                #to do:  fix completeness calculation!
                
                
                break

            
            else:
                #print 'keep going:', h, n, Nold
                Nold = sum(mask)



    #get rid of anything that doesn't meet the richness requirement:
    paredCluster = cuts(paredCluster, richcut = richcut)

    if folds == True: #test and train catalog flags, plus fold:
        paredCluster = testTrainAndFold(paredCluster)

    print 'saving pared clusters:', paredClusterFile
    np.save(paredClusterFile, paredCluster)
        
    return paredCluster

#def calculateNrot(galaxies, mfmin = 14, mfmax = 15.5, mfbin = 0.1):

    #gives a good first guess at how many rotations we need of each cluster

    
    #how many mass bins?
#    numbin = int((mfmax-mfmin)/mfbin)

    #how many clusters are ultimately needed per mass bin?  
#    Npermassbin = flatMF/numbin
    
#    for m in np.linspace(mfmin, mfmax, numbin+1):
#        ind = np.argwhere((galaxies['usualsuspect']==1) & (galaxies['subhalomass']>=10**m) &\
#                          (galaxies['subhalomass']<10**(m+mfbin)))
#        if len(ind) > 0:
#            if len(ind) > 1.2 * Npermassbin/3.:
#                # 20% leeway - we probably just need 3 rotations
#                N = 3
#            else:
#                N = int(Npermassbin/float(len(ind)))+2
#            galaxies['Nrot'][ind] = N
#        
#    for n in range(Nrot):
#        print 'Nrot >', n, ': ', sum(galaxies['Nrot']>=n)
#    return galaxies['Nrot']

                                 
    

def testTrainAndFold(cluster,  mfmin = 14, mfmax = 15.5, mfbin = 0.1):

    #fold = randomly divided among 10:
    cluster['fold'] = cluster['hostid']%Nfold

    #train catalog = flat mass function:
    numbin = int( (mfmax-mfmin)/mfbin)
    numperbin = int(flatMF/float(Nfold)/float(numbin))
    print 'numperbin:', numperbin
    for n in range(Nfold):

        for m in np.linspace(mfmin, mfmax, numbin+1):
            ind = np.argwhere((cluster['Mtot']>=10**m) & (cluster['Mtot']<10**(m+mfbin)) & \
                                  (cluster['fold'] == n))
            numinbin = len(ind)
            #print 'intrain check:', cluster['intrain'][which][0:numperbin]
            if numperbin < numinbin:
                cluster['intrain'][ind[0:numperbin]] = 1
            else:
                cluster['intrain'][ind] = 1

            print n, m, sum(cluster['intrain'])

    #set the first 3 rotations to the 'test catalog'
    testind = np.argwhere(cluster['rotation']<3)
    cluster['intest'][testind] = 1
    print 'testind check', len(testind)
    
    print 'intrain check:', sum(cluster['intrain'])
    print 'intest check:', sum(cluster['intest'])
    return cluster


def updatePLpreds(clusters, m, b, filename):
    print 'update PLpreds'
    clusters['PL1pred'] = (np.log10(clusters['sigmav'])-b)/m
    np.save(filename, clusters)
    return clusters



def main():
    t0 = time.clock()


    #output to a file so that you can see what's going on:
    if outputtoFile == True:
        print 'output text to file'
        sys.stdout = open(txtout, 'w', 1)


    #make a new catalog of cylinders:
    if makeNewCylCat == True:
        #make a new cylinder
        print 'running a new cylinder catalog...'
        makeCylCat(t0)
        print 'finished making new cylinder catalog in ', (time.clock()-t0)/60., 'minutes.'
        print ''

    #load the cylinder file:
    print 'loading cylinder file...',
    clusters = np.load(clusterFile)
    print 'finished in ', (time.clock()-t0)/60., 'minutes.'
    print ''

    #run some checks on the cylinder file
    #print 'running checks on the cylinder file...',
    #checkCyl(clusters, 14.5, 20, 'cylinder')
    #checkMsig(clusters, 14.5, 20, 'cylinder')
    #print 'finished in ', (time.clock()-t0)/60., 'minutes.'
    #print ''

    if clip == 'sigclip':
        #apply a sigma clipping to the cylinders
        # pare the cylinder file
        print 'paring...',
        sigPare(clusters, sigclip)
        paredclusters = np.load(paredClusterFile)
        print 'finished in ', (time.clock()-t0)/60., 'minutes.'
        print ''

        # run some checks on the cylinder file
        #print 'running checks on the cylinder file...',
        #checkCyl(paredclusters, 14.5, 20, '2sigclip')
        #m, b = checkMsig(paredclusters, 14.5, 20, '2sigclip')
        #paredclusters = updatePLpreds(paredclusters, m, b, paredClusterFile)
        #print 'finished in ', (time.clock()-t0)/60., 'minutes.'
        #print ''

    elif clip == 'caustic':
        # this one's not finished yet :)
        print 'no caustic cuts.'

    print 'mischief managed.'

if __name__ == "__main__":
    main()
