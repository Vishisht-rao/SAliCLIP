from Lib_imports import *
from Parameters import *
from Data_pipeline import *

class ImageEncoder(nn.Module):
    def __init__(self, UseGAP=True):
        super(ImageEncoder, self).__init__()
        self.UseGAP = UseGAP

        self.ImageEncoder = clip.load("ViT-B/16", device=device)[0].encode_image

    def forward(self, Images):
        EncoderOutputs = self.ImageEncoder(Images)
        return EncoderOutputs

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class AudioEncoder(nn.Module):
    def __init__(self, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, audioset_pretrain=True, verbose=True, UseGAP=True):

        super(AudioEncoder, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        self.UseGAP = UseGAP

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('AudioSet pretraining: {:s}'.format(str(audioset_pretrain)))
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if audioset_pretrain == False:
            
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=True)
            
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            print(self.v.pos_embed.shape, [1, self.original_num_patches, self.original_embedding_dim])
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            if t_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
            if f_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

        elif audioset_pretrain == True:
            if os.path.exists('audioset_10_10_0.4593.pth') == False:
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='audioset_10_10_0.4593.pth')
            sd = torch.load('audioset_10_10_0.4593.pth', map_location=device)
            audio_model = AudioEncoder(fstride=10, tstride=10, input_fdim=128, input_tdim=1024, audioset_pretrain=False, verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        
        x = self.v.norm(x)

        if self.UseGAP:
            x = torch.mean(x, 1)
        else:
            x = (x[:, 0] + x[:, 1]) / 2

        return x
    
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim=EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CrossProjectionMLPHead(nn.Module):
    def __init__(
        self,
        units=[CROSS_PROJECTION_DIM for _ in range(3)],
        activation_func=nn.ReLU(),
        dropout=0.1
    ):
        super().__init__()
        self.activation_func = activation_func
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.activation_func)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)

    def forward(self, x):
        x = self.sequential(x)
        return x
    
class EncoderModel(pl.LightningModule):
    def __init__(
        self, 
        UseGAPImage=UseGAPImage, 
        UseGAPAudio=UseGAPAudio, 
        UseImageProjectionHead=UseImageProjectionHead,
        UseAudioProjectionHead=UseAudioProjectionHead,
        Temperature=Temperature, 
        FreezeImageEncoder=FreezeImageEncoder, 
        FreezeAudioEncoder=FreezeAudioEncoder,
        embedding_dim=EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM,
        LossFunctionName=LossFunctionName,
        ImageEncoderLearningRate=ImageEncoderLearningRate,
        AudioEncoderLearningRate=AudioEncoderLearningRate,
        ProjectionHeadLearningRate=ProjectionHeadLearningRate,
        Momentum=Momentum,
        WeightDecay=WeightDecay,
        precompute_flag=False
    ):

        super().__init__()
        self.save_hyperparameters(ignore=["precompute_flag"])
        self.precompute_flag = precompute_flag

        self.ImageEncoder = ImageEncoder(UseGAP=UseGAPImage)
    
        if self.hparams.FreezeImageEncoder :
            self.ImageEncoder.eval()
            for param in self.ImageEncoder.parameters():
                param.requires_grad = False


        self.AudioEncoder = AudioEncoder(UseGAP=UseGAPAudio, input_fdim=INPUT_FDIM, input_tdim=INPUT_TDIM)

        if self.hparams.FreezeAudioEncoder :
            self.AudioEncoder.eval()
            for param in self.AudioEncoder.parameters():
                param.requires_grad = False

        self.ImageHead = ProjectionHead(embedding_dim=self.hparams.embedding_dim, projection_dim=self.hparams.projection_dim)
        self.AudioHead = ProjectionHead(embedding_dim=self.hparams.embedding_dim, projection_dim=self.hparams.projection_dim)

        self.cosineSim = nn.CosineSimilarity(dim=1)

        self.image_names_list = []
        self.image_to_audio_map = {}

    def ComputeSimilarityMatrix(self, FinalEncodedImages, FinalEncodedAudios):

        image_embeddings_n = nn.functional.normalize(FinalEncodedImages, p=2, dim=-1)
        audio_embeddings_n = nn.functional.normalize(FinalEncodedAudios, p=2, dim=-1)
        dot_similarity = audio_embeddings_n @ image_embeddings_n.T
        
        return dot_similarity

    def forward(self, Images, Audios):
        FinalEncodedImages = self.ImageEncoder(Images)
        FinalEncodedAudios = self.AudioEncoder(Audios)
        
        if self.hparams.UseImageProjectionHead:
            FinalEncodedImages = self.ImageHead(FinalEncodedImages)
        if self.hparams.UseAudioProjectionHead:
            FinalEncodedAudios = self.AudioHead(FinalEncodedAudios)

        if self.hparams.LossFunctionName == "ClipLoss":
            return FinalEncodedImages, FinalEncodedAudios, None, None
        elif self.hparams.LossFunctionName == "ClipLossCross":
            TransformedImages = self.ImageTransformCross(FinalEncodedImages)
            TransformedAudios = self.AudioTransformCross(FinalEncodedAudios)
            return FinalEncodedImages, FinalEncodedAudios, TransformedImages, TransformedAudios
        else:
            return FinalEncodedImages, FinalEncodedAudios, None, None

    def SymmetricCrossEntropy(self, logits, targets):
        audioLoss = nn.functional.cross_entropy(logits, targets)
        imageLoss = nn.functional.cross_entropy(logits.T, targets.T)
        loss =  (imageLoss + audioLoss) / 2.0 
        return loss.mean()

    def LossFunc(self, image_features, audio_features):
        predY = (audio_features @ image_features.T) / self.hparams.Temperature
        trueY = torch.eye(predY.shape[0])
        trueY = trueY.type_as(predY)
        loss = self.SymmetricCrossEntropy(predY, trueY)
        return loss

    def training_step(self, train_batch, batch_idx):
        images, audios, image_file_names, audio_file_names = train_batch
        FinalEncodedImages, FinalEncodedAudios, TransformedImages, TransformedAudios = self.forward(images, audios)
        
        similarityMatrix = self.ComputeSimilarityMatrix(FinalEncodedImages, FinalEncodedAudios)
        similarities = self.cosineSim(FinalEncodedImages, FinalEncodedAudios)
        mean_similarity = torch.mean(similarities)

        if self.hparams.LossFunctionName == "ClipLoss":
            loss = self.LossFunc(FinalEncodedImages, FinalEncodedAudios)
        elif self.hparams.LossFunctionName == "ClipLossCross":
            loss = (
                self.LossFunc(TransformedImages, FinalEncodedAudios)
                + self.LossFunc(FinalEncodedImages, TransformedAudios)
                ) / 2
        else:
            assert False
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=audios.shape[0])
        self.log('train_similarity', mean_similarity, on_step=True, on_epoch=True, prog_bar=True, batch_size=audios.shape[0])
        return loss

    def validation_step(self, val_batch, batch_idx):
        images, audios, image_file_names, audio_file_names = val_batch
        FinalEncodedImages, FinalEncodedAudios, TransformedImages, TransformedAudios = self.forward(images, audios)

        similarityMatrix = self.ComputeSimilarityMatrix(FinalEncodedImages, FinalEncodedAudios)
        similarities = self.cosineSim(FinalEncodedImages, FinalEncodedAudios)
        mean_similarity = torch.mean(similarities)

        if self.hparams.LossFunctionName == "ClipLoss":
            loss = self.LossFunc(FinalEncodedImages, FinalEncodedAudios)
        elif self.hparams.LossFunctionName == "ClipLossCross":
            loss = (
                self.LossFunc(TransformedImages, FinalEncodedAudios)
                + self.LossFunc(FinalEncodedImages, TransformedAudios)
                ) / 2
        else:
            assert False

        if self.precompute_flag:
            FinalEncodedImageNorm = nn.functional.normalize(FinalEncodedImages, p=2, dim=1)

            for bi in range(audios.shape[0]):
                image_name = image_file_names[bi]
                audio_name = audio_file_names[bi]
                if image_name not in self.image_names_list:
                    self.image_names_list.append(image_name)
                    if hasattr(self, 'FinalEncodedImagesNorm'):
                        self.FinalEncodedImagesNorm = torch.cat((self.FinalEncodedImagesNorm, FinalEncodedImageNorm[bi:bi+1, :]))
                    else:
                        self.FinalEncodedImagesNorm = FinalEncodedImageNorm[bi:bi+1, :]
                if image_name not in self.image_to_audio_map:
                    self.image_to_audio_map[image_name] = FinalEncodedAudios[bi].unsqueeze(0)
                else:
                    self.image_to_audio_map[image_name] = torch.cat((self.image_to_audio_map[image_name], FinalEncodedAudios[bi].unsqueeze(0)), dim=0)


        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=audios.shape[0])
        self.log('val_similarity', mean_similarity, on_step=True, on_epoch=True, prog_bar=True, batch_size=audios.shape[0])
        return loss


    def predict_step(self, pred_batch, batch_idx):
        images, audios, image_file_names, audio_file_names = pred_batch
        FinalEncodedAudios = self.AudioEncoder(audios)
        if self.hparams.UseAudioProjectionHead:
            FinalEncodedAudios = self.AudioHead(FinalEncodedAudios)
        FinalEncodedAudiosNorm = nn.functional.normalize(FinalEncodedAudios, p=2, dim=1)

        similarities = FinalEncodedAudiosNorm @ self.FinalEncodedImagesNorm.T

        
        top_matches, top_indices = torch.topk(similarities, TopMatches)
        similarities, _ = torch.sort(similarities)

        for bi in range(audios.shape[0]):
            image = images[bi]
            audio = audios[bi]
            image_file_name = image_file_names[bi]
            audio_file_name = audio_file_names[bi]
            similarities_row = similarities[bi]
            top_matches_row = top_matches[bi]
            top_indices_row = top_indices[bi]

            IPython.display.Audio(audio_file_name)

            if image_file_name in self.image_names_list:
                img = Image.open(image_file_name)
                display(img)
                print(image_file_name)
            else:
                print("Image not found in dataset")

            print("Max similarity found =", similarities_row[0])
            print("Min similarity found =", similarities_row[-1])

            rows = 3
            cols = 3
            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15,15))

            print("Top " + str(TopMatches) + " Images for the audio" )
            ImgCnt = 0
            LenMatches = top_matches_row.shape[0]
            for i in range(rows):
                for j in range(cols):
                    if ImgCnt < LenMatches:
                        img = Image.open(self.image_names_list[top_indices_row[ImgCnt]])
                        axes[i, j].imshow(img)
                        axes[i, j].set_title("Similarity: " + str(top_matches_row[ImgCnt].item()))
                        ImgCnt += 1
            plt.show()
    
    def configure_optimizers(self):
        Params = [
            {"params": self.ImageEncoder.parameters(), "lr": self.hparams.ImageEncoderLearningRate},
            {"params": self.AudioEncoder.parameters(), "lr": self.hparams.AudioEncoderLearningRate},
            {"params": self.ImageHead.parameters(), "lr": self.hparams.ProjectionHeadLearningRate},
            {"params": self.AudioHead.parameters(), "lr": self.hparams.ProjectionHeadLearningRate},
        ]

        if OptimizerName == "SGD":
            optimizer = torch.optim.SGD(Params, weight_decay=self.hparams.WeightDecay*0.1, momentum=Momentum, nesterov=True)
        elif OptimizerName == "AdamW":
            optimizer = torch.optim.AdamW(Params, weight_decay=self.hparams.WeightDecay, capturable=True) #put momentum
        else:
            assert False

        if SchedulerName == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=Factor,
                patience=Patience,
                min_lr=1e-6,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"
            }
        elif SchedulerName == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=Gamma)
            return [optimizer], [scheduler]
        else:
            assert False

    def compute_additional_metrics(self):
        if not self.image_to_audio_map:
            print("Run validation before computing metrics")
            return
        
        m2, m3 = 0, 0
        for k, v in self.image_to_audio_map.items():
            simMatrix = self.ComputeSimilarityMatrix(v, v)
            m2 += (torch.sum(simMatrix).item()-torch.sum(torch.diagonal(simMatrix)).item())/20
            m3 += torch.sqrt(torch.linalg.det(torch.matmul(v, torch.transpose(v, 0, 1)))).item()

        m2 = m2/len(self.image_to_audio_map.keys())
        m3 = m3/len(self.image_to_audio_map.keys())
        return m2, m3