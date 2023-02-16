from Lib_imports import *
from Parameters import *
from Data_pipeline import *
from Models import *
import streamlit as st
import time

sys.path.append('./taming_transformers')
if os.path.exists('epoch=09-val_loss=1.93--COCO--6054cf6f71f34bc9ae864a780df799e7.ckpt'):
    SAliCLIP = EncoderModel.load_from_checkpoint("epoch=09-val_loss=1.93--COCO--6054cf6f71f34bc9ae864a780df799e7.ckpt")
    SAliCLIP = SAliCLIP.eval().requires_grad_(False).to(device)
    brisque_loss = pyiqa.create_metric('brisque', device=device, as_loss=True)
else :
    print("Model not found in the directory. Please download the model and add it to the directory")
    quit()

def getFbankForAudio(audio_file_path):

    audio_conf_train = {'num_mel_bins': INPUT_FDIM, 'target_length': INPUT_TDIM, 'freqm': FREQM, 'timem': TIMEM, 
                      'mean':5, 'std':5, 'noise':False, 
                      'image_base_dir':IMAGE_DIRECTORY_COCO, 'audio_base_dir':AUDIO_DIRECTORY_COCO, 
                      'frame_length': FRAME_LENGTH, 'frame_shift': FRAME_SHIFT, 'dataLen':CocoTrainLen}
    
    preprocessor = AbstractDataset(audio_conf = audio_conf_train)
    fbank = preprocessor._processAudioFile(audio_file_path)

    return fbank

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))
 
def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()
 
def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]
 
def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size
 
    input = input.view([n * c, 1, h, w])
 
    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])
 
    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])
 
    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)
 
class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward
 
    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)
 
 
replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)
 
    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None
 
 
clamp_with_grad = ClampWithGrad.apply
 
    
def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)
 
    
class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))
 
    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
 
 
def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])
 

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2,p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7))
        self.noise_fac = 0.1
 
 
    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch
 
 
def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model
 

def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

def splitAudio(file_path, extension):
    audio = AudioSegment.from_file(file_path + "." + extension)
    lengthOfFileMS = audio.duration_seconds * 1000
    halfLength = lengthOfFileMS // 2

    delta = max(halfLength // 4, 100)

    firstHalf = audio[:min(halfLength + delta, lengthOfFileMS)]
    secondHalf = audio[max(halfLength - delta, 0):]

    firstHalf.export(file_path + "_1." + extension , format=extension)
    secondHalf.export(file_path + "_2." + extension , format=extension)

    return [file_path + "_1." + extension, file_path + "_2." + extension, file_path + "." + extension]

def GenSetup(prompt, PrevPromptImage, Gitr, BreakAudio, Brisque, mxitr, stepSize, cuts, mybar, tol, prog_p):
    prompts = prompt 
    width =  512
    height =  512
    model = "coco" 
    display_frequency =  50
    if PrevPromptImage == "":
        initial_image = "None"
    else:
        initial_image = PrevPromptImage
    target_images = ""
    seed = -1
    max_iterations = mxitr
    input_images = ""

    model_names={"vqgan_imagenet_f16_16384": 'ImageNet 16384',"vqgan_imagenet_f16_1024":"ImageNet 1024", 
                      "wikiart_16384":"WikiArt 16384", "coco":"COCO-Stuff", "faceshq":"FacesHQ", "sflckr":"S-FLCKR"}
    model_name = model_names[model]     

    if seed == -1:
        seed = None
    if initial_image == "None":
        initial_image = None
    if target_images == "None" or not target_images:
        target_images = []
    else:
        target_images = target_images.split("|")
        target_images = [image.strip() for image in target_images]

    if initial_image or target_images != []:
        input_images = True

   
    if BreakAudio:
        file_name, extension = prompts.split(".")
        print("File name is ", file_name)
        print("Extension is", extension)

        prompts = splitAudio(file_name, extension)
    else:
        prompts = [frase.strip() for frase in prompts.split("|")]
    print("Prompts are", prompts)
    if prompts == ['']:
        prompts = []


    args = argparse.Namespace(
        prompts=prompts,
        image_prompts=target_images,
        noise_prompt_seeds=[],
        noise_prompt_weights=[],
        size=[width, height],
        init_image=initial_image,
        init_weight=0.,
        clip_model='ViT-B/32',
        vqgan_config=f'{model}.yaml',
        vqgan_checkpoint=f'{model}.ckpt',
        step_size=stepSize,
        cutn=cuts,
        cut_pow=1.,
        display_freq=display_frequency,
        seed=seed,
    )
    print("#" * 30)
    
    Gitr = Gen(args, prompts, target_images, max_iterations, Gitr, BreakAudio, Brisque, mybar, tol, prog_p)
    
    print("#" * 30)
    
    return Gitr
    

def Gen(args, prompts, target_images, max_iterations, Gitr, BreakAudio, Brisque, mybar, tol, prog_p):
    print('Using device:', device)
    if prompts:
        print('Using audio prompt:', prompts)
    if target_images:
        print('Using image prompts:', target_images)
    if args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    print('Using seed:', seed)

    model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)

    cut_size = 224

    e_dim = model.quantize.e_dim
    f = 2**(model.decoder.num_resolutions - 1)
    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
    n_toks = model.quantize.n_e
    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]


    if args.init_image:
        pil_image = Image.open(args.init_image).convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=args.step_size)

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

    pMs = []

    for prompt in args.prompts:
        audio_file_path, weight, stop = parse_prompt(prompt)
        audio = getFbankForAudio(audio_file_path)
        audio = torch.unsqueeze(audio, 0).to(device)
        embed = SAliCLIP.AudioEncoder(audio)
        if SAliCLIP.hparams.UseAudioProjectionHead:
            embed = SAliCLIP.AudioHead(embed)
        embed = nn.functional.normalize(embed, p=2, dim=-1)
        pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in args.image_prompts:
        path, weight, stop = parse_prompt(prompt)
        img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = SAliCLIP.ImageEncoder(normalize(batch)).float()
        if SAliCLIP.hparams.UseImageProjectionHead:
            embed = SAliCLIP.ImageHead(embed)
        embed = nn.functional.normalize(embed, p=2, dim=-1)
        pMs.append(Prompt(embed, weight, stop).to(device))


    def synth(z):
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)
    
    @torch.no_grad()
    def checkin(i, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        out = synth(z)

    def ascend_txt(Gitr):
        out = synth(z)

        PILImagesOfCutouts = torch.Tensor().to(device)
        iii = torch.Tensor().to(device)
        transform = transforms.ToPILImage()

        normalized_cutouts = normalize(make_cutouts(out))
        
        iii = SAliCLIP.ImageEncoder(normalized_cutouts)
        if SAliCLIP.hparams.UseImageProjectionHead:
            iii = SAliCLIP.ImageHead(iii)
        iii = nn.functional.normalize(iii, p=2, dim=-1)

        result = []

        if args.init_weight:
            result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)

        for prompt in pMs:
            result.append(prompt(iii))
            
        if Brisque:
           
            result.append(torch.mean(brisque_loss(normalized_cutouts)))
            
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
        img = np.transpose(img, (1, 2, 0))

        filename = f"steps/{Gitr:04}.png"
        imageio.imwrite(filename, np.array(img))
         
        
        if BreakAudio:
            leftHalfWeight = 0.25
            rightHalfWeight = 0.25
            result[0] *= leftHalfWeight
            result[1] *= rightHalfWeight
            
        if Brisque:
            BrisqueWeight = 0.005 / 5
            result[-1] *= BrisqueWeight
            
       
            
        return result

    def train(i, Gitr):
        opt.zero_grad()
        lossAll = ascend_txt(Gitr)
        
        if i % args.display_freq == 0:
            checkin(i, lossAll)
        loss = sum(lossAll)
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.copy_(z.maximum(z_min).minimum(z_max))

    i = 0
    try:
        while True:
            train(i, Gitr)
            mybar.progress(int((Gitr / tol) * 100))
            prog_p.write(str(int((Gitr / tol) * 100)) + "%")
            if i == max_iterations:
                break
            i += 1
            Gitr += 1
    except Exception as e:
        print(e)
        quit()
        
    return Gitr
