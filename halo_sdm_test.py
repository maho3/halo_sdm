
import halo_sdm


config = {
    'scenario': 'test',

    'trainFile' : '/share/scratch1/cmhicks/mattho/debugcat.npy',
    'unlabeledFile': None,

    'seed': 73261,
    'sdrModel': 'NuSDR',
    'n_proc': 16,
    'divfunc': 'kl',
    'K' : 4,
    'unlabeledFile' : None,
    'logFolder' : '/home/mho1/scripts/log/',
    'outFolder' : '/home/mho1/scripts/out/',
    
    'redshift' : '0.117',
    'featureType' : 'MLv',
    'Nfolds' : 10,
    'stand': True
    
    
}


halo_sdm.set_params(config)

halo_sdm.run()


'''
DEFAULT 

redshift = '0.117'
#trainFile = '/share/scratch1/cmhicks/mattho/MDPL2_Maccgeq1e12_small_norecenter_z=' + redshift +'.npy' #training file
trainFile = '/share/scratch1/cmhicks/mattho/debugcat.npy'
#trainFile = '/home/mho1/data/MDPL2_Maccgeq1e12_small_norecenter_z=' + redshift + '_vfiltscale.npy'

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


IN HALO SDM

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

'''
