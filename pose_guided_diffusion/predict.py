import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from tensorfn import load_config
import numpy as np
from config import DiffusionConfig
import torch.distributed as dist
import os, glob, cv2, time, shutil
from models.unet_autoenc import BeatGANsAutoencConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
import torchvision.transforms as transforms
import torchvision

def get_conf():

    return BeatGANsAutoencConfig(image_size=256, 
    in_channels=3+20, 
    model_channels=128, 
    out_channels=3*2,  # also learns sigma
    num_res_blocks=2, 
    num_input_res_blocks=None, 
    embed_channels=512, 
    attention_resolutions=(32, 16, 8,), 
    time_embed_channels=None, 
    dropout=0.1, 
    channel_mult=(1, 1, 2, 2, 4, 4), 
    input_channel_mult=None, 
    conv_resample=True, 
    dims=2, 
    num_classes=None, 
    use_checkpoint=False,
    num_heads=1, 
    num_head_channels=-1, 
    num_heads_upsample=-1, 
    resblock_updown=True, 
    use_new_attention_order=False, 
    resnet_two_cond=True, 
    resnet_cond_channels=None, 
    resnet_use_zero_module=True, 
    attn_checkpoint=False, 
    enc_out_channels=512, 
    enc_attn_resolutions=None, 
    enc_pool='adaptivenonzero', 
    enc_num_res_block=2, 
    enc_channel_mult=(1, 1, 2, 2, 4, 4, 4), 
    enc_grad_checkpoint=False, 
    latent_net_conf=None)

class Predictor():
    def __init__(self):
        """Load the model into memory to make running multiple predictions efficient"""
        #opt = Config('./config/fashion_256.yaml')
        conf = load_config(DiffusionConfig, "config/fashion.conf", show=False)
        #val_dataset, train_dataset = deepfashion_data.get_train_val_dataloader(opt.data, labels_required = True, distributed=False)

        ckpt = torch.load("checkpoints/last.pt")
        self.model = get_conf().make_model()
        self.model.load_state_dict(ckpt["ema"])
        self.model = self.model.cuda()
        self.model.eval()

        self.betas = conf.diffusion.beta_schedule.make()
        self.diffusion = create_gaussian_diffusion(self.betas, predict_xstart = False)#.to(device)
        
        self.pose_list = glob.glob('data/deepfashion_256x256/target_pose/*.npy')
        self.transforms = transforms.Compose([transforms.Resize((256,256), interpolation=Image.BICUBIC),
                            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])
    def predict_pose(
        self,
        image,
        num_poses=1,
        sample_algorithm='ddim',
        nsteps=100,

        ):
        """Run a single prediction on the model"""

        src = Image.open(image)
        src = self.transforms(src).unsqueeze(0).cuda()
        tgt_pose = torch.stack([transforms.ToTensor()(np.load(ps)).cuda() for ps in np.random.choice(self.pose_list, num_poses)], 0)

        src = src.repeat(num_poses,1,1,1)




        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, tgt_pose], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, tgt_pose])
            samples = xs[-1].cuda()


        samples_grid = torch.cat([src[0],torch.cat([samps for samps in samples], -1)], -1)
        samples_grid = (torch.clamp(samples_grid, -1., 1.) + 1.0)/2.0
        pose_grid = torch.cat([torch.zeros_like(src[0]),torch.cat([samps[:3] for samps in tgt_pose], -1)], -1)

        output = torch.cat([1-pose_grid, samples_grid], -2)

        numpy_imgs = output.unsqueeze(0).permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        Image.fromarray(fake_imgs[0]).save('output.png')


    def predict_appearance(
        self,
        image,
        ref_img,
        ref_mask,
        ref_pose,
        sample_algorithm='ddim',
        nsteps=100,

        ):
        """Run a single prediction on the model"""

        src = Image.open(image)
        src = self.transforms(src).unsqueeze(0).cuda()
        
        ref = Image.open(ref_img)
        ref = self.transforms(ref).unsqueeze(0).cuda()

        mask = transforms.ToTensor()(Image.open(ref_mask)).unsqueeze(0).cuda()
        pose =  transforms.ToTensor()(np.load(ref_pose)).unsqueeze(0).cuda()


        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, pose, ref, mask], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, pose, ref, mask], diffusion=self.diffusion)
            samples = xs[-1].cuda()


        samples = torch.clamp(samples, -1., 1.)

        output = (torch.cat([src, ref, mask*2-1, samples], -1) + 1.0)/2.0

        numpy_imgs = output.permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        Image.fromarray(fake_imgs[0]).save('output.png')


ref_img = "data/deepfashion/target_edits/reference_img_100.png"
ref_mask = "data/deepfashion/target_mask/lower/reference_mask_100.png"
ref_pose = "data/deepfashion/target_pose/reference_pose_100.npy"
src_img = "data/deepfashion/target_edits/source_img_93.png"

obj = Predictor()

obj.predict_pose(image='test.jpg', num_poses=4, sample_algorithm = 'ddim',  nsteps = 50)

obj.predict_appearance(image='test.jpg', ref_img = ref_img, ref_mask = ref_mask, ref_pose = ref_pose, sample_algorithm = 'ddim',  nsteps = 50)
