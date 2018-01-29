# Usage: a collection of cosmology and cluster tools

# cosmoparams should be of the form:
#         cosmoparams = {'OmegaM':0.3121,'OmegaL':0.6879,'Omegak':0.,'hubble':0.6751}

# assumes the following import statements:
#         import numpy as np



G = 4.3016e-9 #Mpc/h km^2 (Msolar/h)^(-1) s^(-2)



############################## DISTANCE TOOLS ###############################


def invE(z, **cosmoparams):
    #function we need to integrate to solve for the comoving distance!
    return 1./np.sqrt(cosmoparams['OmegaM'] * (1+z)**3 + cosmoparams['Omegak'] * (1+z)**2 + \
                      cosmoparams['OmegaL'])

def integrate_invE(z, **cosmoparams):
    def func(z):
        return invE(z, **cosmoparams)
    return integrate.quad(func, 0, z)[0]


def Esquared(z, **cosmoparams):
    return cosmoparams['OmegaM']*(1.+z)**3 + cosmoparams['OmegaL']

def DA(z, **cosmoparams):
    #angular diameter distance to z in Mpc
    hubbleDist = hubbleDistance(**cosmoparams)
    
    if cosmoparams['Omegak'] > 0:
        comovingDist = hubbleDist / np.sqrt(cosmoparams['Omegak']) *\
                       np.sinh(np.sqrt(cosmoparams['Omegak']) *\
                               z_to_Dc(z, **cosmoparams)/hubbleDist)
    elif cosmoparams['Omegak'] < 0:
        comovingDist = hubbleDist / np.sqrt(cosmoparams['Omegak']) *\
                       np.sin(np.sqrt(cosmoparams['Omegak']) *\
                              z_to_Dc(z, **cosmoparams)/hubbleDist)
    else:
        comovingDist = z_to_Dc(z, **cosmoparams)

    return comovingDist / (1.+z)

def hubbleDistance(**cosmoparams):
    #print 'hubbledist cosmo:', cosmoparams
    return 2.99792e3 / cosmoparams['hubble'] #Mpc, see Hogg 2000 eq 4
          

def z_to_Dc(z, **cosmoparams):
    hubbleDist = hubbleDistance(**cosmoparams)
    comovingDist = np.zeros(len(z), 'f')
    for i, zi in enumerate(z):
        comovingDist[i] = hubbleDist * integrate_invE(zi, **cosmoparams)
    return comovingDist

def distance_to_z(Dc, zmin = 0.0, zmax = 2, **cosmoparams):
    #takes a comoving distance, returns a redshift
    hubbleDist = hubbleDistance(**cosmoparams)
    z = np.linspace(zmin, zmax, 1000)
    comovingDist = z_to_Dc(z, **cosmoparams)
    f = scipy.interpolate.interp1d(comovingDist, z)
    return f(Dc)

def makeTable_Dc_to_z(zmax=1.5, N=7000, **cosmoparams):
    #make a lookup table that converts back and forth between distance and redshift
    #it's best to just really overshoot this table... it's cheap and makes life easier.
    hubbleDist = hubbleDistance(**cosmoparams)
    dat = np.zeros(N, dtype=[('z', 'f'), ('Dc', 'f')])
    dat['z'] = np.linspace(0, zmax, N)
    for n in range(N):
        dat['Dc'][n] = hubbleDist * integrate.quad(invE, 0, dat['z'][n])[0]

    return dat



############################## VELOCITY TOOLS ###############################


def observedVlos(peculiarvelocity, z=None, Dc=None, **cosmoparams):
    #returns observed recessional velocity,
    #given a peculiar velocity and redshift or comoving distance

    if z is not None:
        return addVelocitiesRelativistically(z_to_vlos(z), peculiarvelocity)
    elif Dc is not None:
        return addVelocitiesRelativistically(\
                         z_to_vlos(distance_to_z(Dc, **cosmoparams)), peculiarvelocity)
    else:
        return None
        
    
    
def z_to_vlos(z, units = 'kms'):
    if units == 'kms':
        #units of velocities are in km/s
        c=2.99792458e5 #km/s
    x = (1.+z)**2
    return (x-1.)/(x+1.)*c

def addVelocitiesRelativistically(v1, v2, units = 'kms'):
    #returns v1 plus v2, where these are in km/s (otherwise, need to change the 
    if units == 'kms':
        #units of velocities are in km/s
        c=2.99792458e5 #km/s
    return (v1 + v2)/(1+v1*v2/c**2)

def clusterFrameVlos(z, zcluster, units = 'kms'):
    #takes cluster redshift and numpy array of galaxy redshifts,
    #returns the los velocity WITH RESPECT THE THE CLUSTER REDSHIFT
    if units == 'kms':
        #units of velocities are in km/s
        c=2.99792458e5 #km/s
    vlos = addVelocitiesRelativistically(z_to_vlos(z), -1.* z2vlos(zcluster))
    return vlos

    
############################## CLUSTER TOOLS ###############################


def YSZ_to_M500(YSZ, z, alpha = 1.79, beta = 0.66, siglogY = 0.075, \
                    bias = 0.2, Ystar = 10**(-0.19), **cosmoparams):
    
    print 'ysz cosmo:', cosmoparams['hubble']

    hubbleDist = hubbleDistance(**cosmoparams)

    MSZ = 6e14 / (1-bias) * \
          (  Esquared(z, **cosmoparams)**(-beta/2.)/Ystar * \
             (cosmoparams['hubble']/0.7)**(2.-alpha) * \
             (DA(z, **cosmoparams)/hubbleDist)**2 * \
             YSZ/1e-4                           )**(1./alpha)

    return MSZ


def M500_to_YSZ(M500, z, alpha = 1.79, beta = 0.66, siglogY = 0.075, \
                    bias = 0.2, Ystar = 10**(-0.19), scatter = False, **cosmoparams):
    #calculates Y_SZ from M_500 according to equation 7 from the paper
    #Planck 2013 Results XX Cosmology from Sunyaev-Zeldovich cluster counts
    #and reproduces figure 3
    #this equation expects M in Msolar/h
    #and returns YSZ in Mpc^2/h
    hubbleDist = hubbleDistance(**cosmoparams)

    

    Y500 = Ystar * \
        (cosmoparams['hubble'] / 0.7)**(-2.+alpha) * \
        (((1.-bias)*M500/cosmoparams['hubble'])/(6e14))**alpha * \
        (np.sqrt(Esquared(z, **cosmoparams))**(beta)) * \
        (10**(-4.))/((DA(z, **cosmoparams)/1000.)**2)*cosmoparams['hubble']
    #Note:  my M in Ms/h, but Planck expects Ms
    #Note:  to make Planck's fit agree with Nick Battaglia's, DA has to be in Gpc.  Oy.

    if scatter == False:
        return Y500
    else:
        print 'adding scatter to YSZ'
        return 10**(np.log10(Y500)+np.random.normal(0, siglogY, len(Y500)))

def M500_to_Theta500(M500, z, bias = 0.2, Thetastar = 6.997, **cosmoparams):
    #returns the angular size in arcmins
    #Planck xX 2013 equation 9
    Theta500 = Thetastar * (cosmoparams['hubble']/0.7)**(-2./3.) * \
                       (((1-bias) * M500/cosmoparams['hubble'])/(3e14))**(1./3.) *\
                       Esquared(z, **cosmoparams)**-1./3. * \
                       (500./DA(z, **cosmoparams))
    #Note:  my m is in Ms/h, but Planck expects Ms
    #Note:  for this particular Planck equation, they want DA in Mpc... argh!
    return Theta500


    


