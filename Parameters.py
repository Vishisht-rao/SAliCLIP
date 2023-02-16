import configparser

config = configparser.ConfigParser()
config.read('config')

Paths = config['Paths']
Hyperparameters = config['Hyperparameters']
Device = config['Device']

# Device
device = str(Device['device'])
	
pin_memory = True
if device == "cpu":
    pin_memory = False

# Paths
AUDIO_DIRECTORY_FLICKR = Paths['AUDIO_DIRECTORY_FLICKR']
IMAGE_DIRECTORY_FLICKR = Paths['IMAGE_DIRECTORY_FLICKR'] 
AUDIO_DIRECTORY_COCO = Paths['AUDIO_DIRECTORY_COCO'] 
IMAGE_DIRECTORY_COCO = Paths['IMAGE_DIRECTORY_COCO'] 
AUDIO_DIRECTORY_OBJECTNET = Paths['AUDIO_DIRECTORY_OBJECTNET'] 
IMAGE_DIRECTORY_OBJECTNET = Paths['IMAGE_DIRECTORY_OBJECTNET'] 
PREDICTION_DIR_FLICKR = Paths['PREDICTION_DIR_FLICKR'] 
EXPERIMENTS_DIR = Paths['EXPERIMENTS_DIR']  

# Hyperparameters
UseGAPImage = bool(Hyperparameters['UseGAPImage'])
UseGAPAudio = bool(Hyperparameters['UseGAPAudio'])
UseImageProjectionHead = bool(Hyperparameters['UseImageProjectionHead'])
UseAudioProjectionHead = bool(Hyperparameters['UseAudioProjectionHead'])
EMBEDDING_DIM = int(Hyperparameters['EMBEDDING_DIM'])
PROJECTION_DIM = int(Hyperparameters['PROJECTION_DIM'])
CROSS_PROJECTION_DIM = PROJECTION_DIM if UseImageProjectionHead or UseAudioProjectionHead else EMBEDDING_DIM 
FreezeImageEncoder = bool(Hyperparameters['FreezeImageEncoder'])
FreezeAudioEncoder = bool(Hyperparameters['FreezeAudioEncoder'])
INPUT_TDIM = int(Hyperparameters['INPUT_TDIM'])
INPUT_FDIM = int(Hyperparameters['INPUT_FDIM'])
FRAME_LENGTH = int(Hyperparameters['FRAME_LENGTH'])
FRAME_SHIFT = int(Hyperparameters['FRAME_SHIFT'])
NOISE = bool(Hyperparameters['NOISE'])
FREQM = int(Hyperparameters['FREQM'])
TIMEM = int(Hyperparameters['TIMEM'])
SKIP_NORM = bool(Hyperparameters['SKIP_NORM'])
EXPERIMENT_NAME = str(Hyperparameters['EXPERIMENT_NAME'])
MLFLOW_RUN_ID = None 
DATASET = str(Hyperparameters['DATASET'])
FlickrTrainLen = int(Hyperparameters['FlickrTrainLen'])
FlickrDataLen = int(Hyperparameters['FlickrDataLen'])
FlickrPredictLen = int(Hyperparameters['FlickrPredictLen'])
CocoTrainLen = int(Hyperparameters['CocoTrainLen'])
CocoValLen = int(Hyperparameters['CocoValLen'])
ObjectNetTrainLen = int(Hyperparameters['ObjectNetTrainLen'])
ObjectNetValLen = int(Hyperparameters['ObjectNetValLen'])
ObjectNetTestLen = int(Hyperparameters['ObjectNetTestLen'])
LossFunctionName = str(Hyperparameters['LossFunctionName'])
ImageEncoderLearningRate = float(Hyperparameters['ImageEncoderLearningRate'])
AudioEncoderLearningRate = float(Hyperparameters['AudioEncoderLearningRate'])
ProjectionHeadLearningRate = float(Hyperparameters['ProjectionHeadLearningRate'])
WeightDecay = float(Hyperparameters['WeightDecay'])
Factor = float(Hyperparameters['Factor'])
Patience = int(Hyperparameters['Patience'])
Gamma = float(Hyperparameters['Gamma'])
OptimizerName = str(Hyperparameters['OptimizerName'])
SchedulerName = str(Hyperparameters['SchedulerName'])
NumEpochs = int(Hyperparameters['NumEpochs'])
BATCH_SIZE = int(Hyperparameters['BATCH_SIZE'])
Temperature = float(Hyperparameters['Temperature'])
Momentum = float(Hyperparameters['Momentum'])
PRECOMPUTE_FLAG = bool(Hyperparameters['PRECOMPUTE_FLAG'])
TopMatches = int(Hyperparameters['TopMatches'])