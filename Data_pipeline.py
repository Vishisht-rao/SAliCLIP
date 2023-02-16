from Lib_imports import *
from Parameters import *

class AbstractDataset(Dataset):
    def __init__(self, audio_conf, image_transforms=None, ImageFeatureExtractor=None):
        self.files = []
        self.audio_conf = audio_conf
        self.image_transforms = image_transforms
        self.ImageFeatureExtractor = ImageFeatureExtractor
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.frame_shift = self.audio_conf.get('frame_shift')
        self.frame_length = self.audio_conf.get('frame_length')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
       
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        self.noise = self.audio_conf.get('noise')

        self.dataLen = self.audio_conf.get('dataLen')
        self.image_base_dir = audio_conf.get('image_base_dir')
        self.audio_base_dir = audio_conf.get('audio_base_dir')

    def _wav2fbank(self, filename):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
        
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=self.frame_shift, frame_length = self.frame_length)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def _processImageFile(self, image_file):
        image = Image.open(image_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        if image is None:
            return

        if self.ImageFeatureExtractor:
            image = self.ImageFeatureExtractor(image)

        if self.image_transforms:
            image = self.image_transforms(image)

        return image

    def _processAudioFile(self, audio_file):
        fbank = self._wav2fbank(audio_file)

        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)

        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)

        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        
        return fbank

    @abstractmethod
    def _loadFiles(self):
        pass

    def __getitem__(self, index):
        image_file, audio_file = self.files[index]
        image = self._processImageFile(image_file)
        fbank = self._processAudioFile(audio_file)
        return image, fbank, image_file, audio_file

    def __len__(self):
        return len(self.files)
    
class FlickrDataset(AbstractDataset):
    def __init__(self, audio_conf, image_transforms=None, ImageFeatureExtractor=None):
        super().__init__(audio_conf, image_transforms, ImageFeatureExtractor)
        self._loadFiles()

    def _loadFiles(self):
        audio_file_names = list(os.listdir(self.audio_base_dir))[:self.dataLen]
        self.files = [[self.image_base_dir+audio_file_name[:-6]+".jpg", self.audio_base_dir+audio_file_name] for audio_file_name in audio_file_names]  

class CocoDataset(AbstractDataset):
    def __init__(self, audio_conf, json_path, image_transforms=None, ImageFeatureExtractor=None):
        super().__init__(audio_conf, image_transforms, ImageFeatureExtractor)
        self.json_path = json_path
        self._loadFiles()

    def _loadFiles(self):
        with open(self.json_path) as f:
            data = json.load(f)["data"]

        for item in data:
            for caption in item["captions"]:
                self.files.append([self.image_base_dir+item["image"], self.audio_base_dir+caption["wav"]])
        
        self.files = self.files[:self.dataLen]
        
class ObjectNetDataset(AbstractDataset):
    def __init__(self, audio_conf, json_path, allowed_classes=None, image_transforms=None, ImageFeatureExtractor=None):
        super().__init__(audio_conf, image_transforms, ImageFeatureExtractor)
        self.json_path = json_path
        self.allowed_classes = allowed_classes
        self._loadFiles()

    def _loadFiles(self):
        with open(self.json_path) as f:
            data = json.load(f)["data"]

        for item in data:
            item_class = item["wav"].split("/")[0]
            if self.allowed_classes is None or item_class in self.allowed_classes: 
                self.files.append([self.image_base_dir+'.'.join(item["image"].split(".")[:-1])+".jpg", self.audio_base_dir+"wavs/"+item["wav"]])
        
        self.files = self.files[:self.dataLen]


class FlickrDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
        
        self.flickr_audio_mean = 4.26
        self.flickr_audio_std = 4.57 
        audio_conf = {'num_mel_bins': INPUT_FDIM, 'target_length': INPUT_TDIM, 'freqm': FREQM, 'timem': TIMEM, 
                      'mean':self.flickr_audio_mean, 'std':self.flickr_audio_std, 'noise':NOISE, 
                      'image_base_dir':IMAGE_DIRECTORY_FLICKR, 'audio_base_dir':AUDIO_DIRECTORY_FLICKR, 
                      'frame_length': FRAME_LENGTH, 'frame_shift': FRAME_SHIFT, 'dataLen':FlickrDataLen}

        self.ImageFeatureExtractor = clip.load("ViT-B/16", device=device)[1]
        
        image_transforms = transforms.RandomOrder([
            transforms.RandomHorizontalFlip(p=0.05),
            transforms.RandomPerspective(p=0.01, distortion_scale=0.2),
            transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(degrees=30)]), p=0.05),
            ])

        self.flickr_full = FlickrDataset(audio_conf=audio_conf, ImageFeatureExtractor=self.ImageFeatureExtractor, image_transforms=image_transforms)
        self.flickr_train, self.flickr_val = random_split(self.flickr_full, [FlickrTrainLen, len(self.flickr_full)-FlickrTrainLen])

    def GetNormStats(audio_conf):
        train_loader = torch.utils.data.DataLoader(
            FlickrDataset(audio_conf=audio_conf), batch_size=128, shuffle=False, num_workers=2, pin_memory=pin_memory)
        mean=[]
        std=[]
        for i, image_input, audio_input, image_file_input, audio_file_input in enumerate(train_loader):
            cur_mean = torch.mean(audio_input[0])
            cur_std = torch.std(audio_input[0])
            mean.append(cur_mean)
            std.append(cur_std)

        print(np.mean(mean), np.mean(std))
        return np.mean(mean), np.mean(std)

    def train_dataloader(self):
        return DataLoader(self.flickr_train, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=pin_memory)

    def val_dataloader(self):
        return DataLoader(self.flickr_val, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=pin_memory)

    def test_dataloader(self):
        return DataLoader(self.flickr_val, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=pin_memory)

    def predict_dataloader(self):
        audio_conf = {'num_mel_bins': INPUT_FDIM, 'target_length': INPUT_TDIM, 'freqm': 0, 'timem': 0, 
                      'mean':self.flickr_audio_mean, 'std':self.flickr_audio_std, 'noise':False,
                      'image_base_dir':IMAGE_DIRECTORY_FLICKR, 'audio_base_dir':PREDICTION_DIR_FLICKR, 
                      'frame_length': FRAME_LENGTH, 'frame_shift': FRAME_SHIFT, 'dataLen': FlickrPredictLen}
        return DataLoader(FlickrDataset(audio_conf=audio_conf, ImageFeatureExtractor=self.ImageFeatureExtractor), batch_size=1, num_workers=os.cpu_count(), pin_memory=pin_memory)
    

    
class CocoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()

        self.batch_size = batch_size
        
        self.coco_audio_mean = 4.26
        self.coco_audio_std = 4.57
        audio_conf_train = {'num_mel_bins': INPUT_FDIM, 'target_length': INPUT_TDIM, 'freqm': FREQM, 'timem': TIMEM, 
                      'mean':self.coco_audio_mean, 'std':self.coco_audio_std, 'noise':NOISE, 
                      'image_base_dir':IMAGE_DIRECTORY_COCO, 'audio_base_dir':AUDIO_DIRECTORY_COCO, 
                      'frame_length': FRAME_LENGTH, 'frame_shift': FRAME_SHIFT, 'dataLen':CocoTrainLen}

        audio_conf_val = {'num_mel_bins': INPUT_FDIM, 'target_length': INPUT_TDIM, 'freqm': FREQM, 'timem': TIMEM, 
                      'mean':self.coco_audio_mean, 'std':self.coco_audio_std, 'noise':NOISE, 
                      'image_base_dir':IMAGE_DIRECTORY_COCO, 'audio_base_dir':AUDIO_DIRECTORY_COCO, 
                      'frame_length': FRAME_LENGTH, 'frame_shift': FRAME_SHIFT, 'dataLen':CocoValLen}

        self.ImageFeatureExtractor = clip.load("ViT-B/16", device=device)[1]
        
        image_transforms = transforms.RandomOrder([
            transforms.RandomHorizontalFlip(p=0.05),
            transforms.RandomPerspective(p=0.01, distortion_scale=0.2),
            transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(degrees=30)]), p=0.05),
            ])

        self.coco_train = CocoDataset(audio_conf=audio_conf_train, json_path=audio_conf_train["audio_base_dir"]+"SpokenCOCO_train.json", ImageFeatureExtractor=self.ImageFeatureExtractor, image_transforms=image_transforms)
        self.coco_val = CocoDataset(audio_conf=audio_conf_val, json_path=audio_conf_val["audio_base_dir"]+"SpokenCOCO_val.json", ImageFeatureExtractor=self.ImageFeatureExtractor)

    def GetNormStats(audio_conf):
        train_loader = torch.utils.data.DataLoader(
            CocoDataset(audio_conf=audio_conf, json_path=AUDIO_DIRECTORY_COCO+"SpokenCOCO_train.json"), batch_size=128, shuffle=False, num_workers=2, pin_memory=pin_memory)
        mean=[]
        std=[]
        for i, image_input, audio_input, image_file_input, audio_file_input in enumerate(train_loader):
            cur_mean = torch.mean(audio_input[0])
            cur_std = torch.std(audio_input[0])
            mean.append(cur_mean)
            std.append(cur_std)

        print(np.mean(mean), np.mean(std))
        return np.mean(mean), np.mean(std)

    def train_dataloader(self):
        return DataLoader(self.coco_train, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=pin_memory)

    def val_dataloader(self):
        return DataLoader(self.coco_val, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=pin_memory)

    def test_dataloader(self):
        return DataLoader(self.coco_val, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=pin_memory)

    def predict_dataloader(self):
        audio_conf = {'num_mel_bins': INPUT_FDIM, 'target_length': INPUT_TDIM, 'freqm': 0, 'timem': 0, 
                      'mean':self.coco_audio_mean, 'std':self.coco_audio_std, 'noise':False, 'image_base_dir':IMAGE_DIRECTORY_COCO, 
                      'audio_base_dir':AUDIO_DIRECTORY_COCO, 'frame_length': FRAME_LENGTH, 'frame_shift': FRAME_SHIFT, 'dataLen': CocoValLen}
        return DataLoader(CocoDataset(audio_conf=audio_conf, json_path=audio_conf["audio_base_dir"]+"SpokenCOCO_val.json", ImageFeatureExtractor=self.ImageFeatureExtractor), batch_size=1, num_workers=os.cpu_count(), pin_memory=pin_memory)
    
    
class ObjectNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()

        self.batch_size = batch_size
        
        self.objectnet_audio_mean = 4.26
        self.objectnet_audio_std = 4.57
        audio_conf_train = {'num_mel_bins': INPUT_FDIM, 'target_length': INPUT_TDIM, 'freqm': FREQM, 'timem': TIMEM, 
                      'mean':self.objectnet_audio_mean, 'std':self.objectnet_audio_std, 'noise':NOISE, 
                      'image_base_dir':IMAGE_DIRECTORY_OBJECTNET, 'audio_base_dir':AUDIO_DIRECTORY_OBJECTNET, 
                      'frame_length': FRAME_LENGTH, 'frame_shift': FRAME_SHIFT, 'dataLen':ObjectNetTrainLen}

        audio_conf_val = {'num_mel_bins': INPUT_FDIM, 'target_length': INPUT_TDIM, 'freqm': FREQM, 'timem': TIMEM, 
                      'mean':self.objectnet_audio_mean, 'std':self.objectnet_audio_std, 'noise':NOISE, 
                      'image_base_dir':IMAGE_DIRECTORY_OBJECTNET, 'audio_base_dir':AUDIO_DIRECTORY_OBJECTNET, 
                      'frame_length': FRAME_LENGTH, 'frame_shift': FRAME_SHIFT, 'dataLen':ObjectNetValLen}

        audio_conf_test = {'num_mel_bins': INPUT_FDIM, 'target_length': INPUT_TDIM, 'freqm': FREQM, 'timem': TIMEM, 
                      'mean':self.objectnet_audio_mean, 'std':self.objectnet_audio_std, 'noise':NOISE, 
                      'image_base_dir':IMAGE_DIRECTORY_OBJECTNET, 'audio_base_dir':AUDIO_DIRECTORY_OBJECTNET, 
                      'frame_length': FRAME_LENGTH, 'frame_shift': FRAME_SHIFT, 'dataLen':ObjectNetTestLen}

        self.ImageFeatureExtractor = clip.load("ViT-B/16", device=device)[1]
        
        image_transforms = transforms.RandomOrder([
            transforms.RandomHorizontalFlip(p=0.05),
            transforms.RandomPerspective(p=0.01, distortion_scale=0.2),
            transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(degrees=30)]), p=0.05),
            ])

        self.objectnet_train = ObjectNetDataset(audio_conf=audio_conf_train, json_path=audio_conf_train["audio_base_dir"]+"metadata/SON-train.json", ImageFeatureExtractor=self.ImageFeatureExtractor, image_transforms=image_transforms)
        self.objectnet_val = ObjectNetDataset(audio_conf=audio_conf_val, json_path=audio_conf_val["audio_base_dir"]+"metadata/SON-val.json", ImageFeatureExtractor=self.ImageFeatureExtractor)
        self.objectnet_test = ObjectNetDataset(audio_conf=audio_conf_test, json_path=audio_conf_test["audio_base_dir"]+"metadata/SON-test.json", ImageFeatureExtractor=self.ImageFeatureExtractor)

    def GetNormStats(audio_conf):
        train_loader = torch.utils.data.DataLoader(
            ObjectNetDataset(audio_conf=audio_conf, json_path=AUDIO_DIRECTORY_OBJECTNET+"metadata/SON-train.json"), batch_size=128, shuffle=False, num_workers=2, pin_memory=pin_memory)
        mean=[]
        std=[]
        for i, image_input, audio_input, image_file_input, audio_file_input in enumerate(train_loader):
            cur_mean = torch.mean(audio_input[0])
            cur_std = torch.std(audio_input[0])
            mean.append(cur_mean)
            std.append(cur_std)

        print(np.mean(mean), np.mean(std))
        return np.mean(mean), np.mean(std)

    def train_dataloader(self):
        return DataLoader(self.objectnet_train, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=pin_memory)

    def val_dataloader(self):
        return DataLoader(self.objectnet_val, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=pin_memory)

    def test_dataloader(self):
        return DataLoader(self.objectnet_test, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=pin_memory)

    def predict_dataloader(self):
        audio_conf = {'num_mel_bins': INPUT_FDIM, 'target_length': INPUT_TDIM, 'freqm': 0, 'timem': 0, 
                      'mean':self.objectnet_audio_mean, 'std':self.objectnet_audio_std, 'noise':False, 
                      'image_base_dir':IMAGE_DIRECTORY_OBJECTNET, 'audio_base_dir':AUDIO_DIRECTORY_OBJECTNET, 
                      'frame_length': FRAME_LENGTH, 'frame_shift': FRAME_SHIFT, 'dataLen': ObjectNetTestLen}
        return DataLoader(ObjectNetDataset(audio_conf=audio_conf, json_path=audio_conf["audio_base_dir"]+"metadata/SON-test.json", ImageFeatureExtractor=self.ImageFeatureExtractor), batch_size=1, num_workers=os.cpu_count(), pin_memory=pin_memory)