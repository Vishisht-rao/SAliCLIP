import sys
from Lib_imports import *
from Parameters import *
from Models import *
from Data_pipeline import *
from Videogen import *
from Helpers import *
from torchmetrics import AveragePrecision

audio_conf_classify = {'num_mel_bins': INPUT_FDIM, 'target_length': INPUT_TDIM, 'freqm': FREQM, 'timem': TIMEM, 
                  'mean':5, 'std':5, 'noise':False, 
                  'image_base_dir':"", 'audio_base_dir':"", 
                  'frame_length': FRAME_LENGTH, 'frame_shift': FRAME_SHIFT, 'dataLen':CocoTrainLen}

ImageFeatureExtractor = clip.load("ViT-B/16", device=device)[1]

preprocessor = AbstractDataset(audio_conf=audio_conf_classify, ImageFeatureExtractor=ImageFeatureExtractor)