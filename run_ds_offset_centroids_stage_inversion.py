"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
import math
from time import perf_counter

import dill
import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage

from tqdm import trange
import dnnlib
import legacy
from metrics import metric_utils
import timm

from training.diffaug import DiffAugment
from pg_modules.blocks import Interpolate
from torch_utils import misc
from losses import BL2Loss    
import models.networks as networks
import options

from torch.optim import Optimizer

def get_morphed_w_code(new_w_code, fixed_w, regularizer_alpha=30):
    interpolation_direction = new_w_code - fixed_w
    interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
    direction_to_move = regularizer_alpha * interpolation_direction / interpolation_direction_norm
    result_w = fixed_w + direction_to_move
    return result_w


def space_regularizer_loss(
    G_pti,
    G_original,
    w_batch,
    vgg16,
    num_of_sampled_latents=1,
    lpips_lambda=10,
    c=None,
):

    z_samples = np.random.randn(num_of_sampled_latents, G_original.z_dim)
    z_samples = torch.from_numpy(z_samples).to(w_batch.device)

    if not G_original.c_dim:
        c_samples = None
    else:
        if c is None:
            c_samples = F.one_hot(torch.randint(G_original.c_dim, (num_of_sampled_latents,)), G_original.c_dim)
            c_samples = c_samples.to(w_batch.device)
        else:
            c_samples = c[0].unsqueeze(0).repeat([num_of_sampled_latents, 1])
            c_samples = c_samples.to(w_batch.device)

    w_samples = G_original.mapping(z_samples, c_samples, truncation_psi=0.5)
    territory_indicator_ws = [get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]

    for w_code in territory_indicator_ws:
        new_img = G_pti.synthesis(w_code, noise_mode='none', force_fp32=True)
        with torch.no_grad():
            old_img = G_original.synthesis(w_code, noise_mode='none', force_fp32=True)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if new_img.shape[-1] > 256:
            new_img = F.interpolate(new_img, size=(256, 256), mode='area')
            old_img = F.interpolate(old_img, size=(256, 256), mode='area')

        new_feat = vgg16(new_img, resize_images=False, return_lpips=True)
        old_feat = vgg16(old_img, resize_images=False, return_lpips=True)
        lpips_loss = lpips_lambda * (old_feat - new_feat).square().sum()

    return lpips_loss / len(territory_indicator_ws)

def loss_geocross(latent, ws):
    if(latent.shape[1] == 1):
        return 0
    else:
        X = latent.view(-1, 1, ws, 512)
        Y = latent.view(-1, ws, 1, 512)
        A = ((X-Y).pow(2).sum(-1)+1e-9).sqrt()
        B = ((X+Y).pow(2).sum(-1)+1e-9).sqrt()
        D = 2*torch.atan2(A, B)
        D = ((D.pow(2)*512).mean((1, 2))/8.).sum()
        return D

def pivotal_tuning(
    G, D, 
    w_pivot, w_offset,
    target,
    device: torch.device,
    num_steps=350,
    learning_rate = 3e-4,
    noise_mode="const",
    verbose = False, c=None
    bl2=False, rank_w=0,
):
    G_original = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    G_pti = copy.deepcopy(G).train().requires_grad_(True).to(device)
    w_pivot.requires_grad_(False)

    # Load VGG16 feature detector.
    vgg16_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
    vgg16 = metric_utils.get_feature_detector(vgg16_url, device=device)

    # l2 criterion
    if bl2:
        l2_criterion = BL2Loss()
    else:
        l2_criterion = torch.nn.MSELoss(reduction='mean')

    if rank_w > 0:
        opt = options.parse('script/train/ranker.yaml', is_train=True)
        netR = networks.define_R(opt).to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    # initalize optimizer
    optimizer = torch.optim.Adam(G_pti.parameters(), lr=learning_rate)

    # run optimization loop
    all_images = []
    ft_num = [8, 16, 24, 32, 38]
    # ft_num = [38, 38, 38, 38, 38]
    for step in range(num_steps):
        # Synth images from opt_w.
        synth_images = G_pti.synthesis(w_pivot+w_offset, noise_mode=noise_mode)

        # track images
        synth_images = (synth_images + 1) * (255/2)
        synth_images_np = synth_images.clone().detach().permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        all_images.append(synth_images_np)
        synth_images_ori = synth_images.clone()

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        gen_im_lr = F.interpolate(synth_images_ori, size=(target_images.shape[-2],target_images.shape[-1]) , mode='bicubic')
        # LPIPS loss
        synth_features = vgg16(gen_im_lr, resize_images=False, return_lpips=True)
        lpips_loss = (target_features - synth_features).square().sum()

        # MSE loss
        mse_loss = l2_criterion(target_images, gen_im_lr)

        # space regularizer
        reg_loss = space_regularizer_loss(G_pti, G_original, w_pivot+w_offset, vgg16, c=c)

        rank_loss = 0
        if rank_w > 0:
            l_g_rank = netR(synth_images)
            l_g_rank = torch.sigmoid(l_g_rank)
            l_g_rank = torch.sum(l_g_rank)
            rank_loss = l_g_rank * rank_w

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss = mse_loss*10 + lpips_loss + reg_loss + rank_loss
        loss.backward()
        optimizer.step()

        msg  = f'[ step {step+1:>4d}/{num_steps}] '
        msg += f'[ loss: {float(loss):<5.2f}] '
        msg += f'[ lpips: {float(lpips_loss):<5.2f}] '
        msg += f'[ mse: {float(mse_loss):<5.4f}] '
        msg += f'[ reg: {float(reg_loss):<5.2f}] '
        msg += f'[ rank: {float(rank_loss):<5.2f}] '
        if verbose: print(msg)

    return all_images, G_pti

def project(
    G, D, offset_reg, class_idx, 
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps = 1000,
    w_avg_samples = 10000,
    initial_learning_rate = 0.1,
    lr_rampdown_length = 0.25,
    lr_rampup_length = 0.05,
    verbose = False,
    device: torch.device,
    noise_mode="const",
    centroids_path: str,
    geo_reg=0,
    center_idx=-1,
    bl2=False,
    rank_w=0,
    ema=False,
    reg_w=0.5,
):
    # assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    if bl2:
        l2_criterion = BL2Loss()
    else:
        l2_criterion = torch.nn.MSELoss(reduction='mean')
    if rank_w > 0:
        opt = options.parse('script/train/ranker.yaml', is_train=True)
        netR = networks.define_R(opt).to(device)
    
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = torch.from_numpy(np.random.RandomState(123).randn(w_avg_samples, G.z_dim)).to(device)

    # get class probas by classifier
    if not G.c_dim:
        c_samples = None
    else:
        if class_idx == -1:
            classifier = timm.create_model('deit_base_distilled_patch16_224', pretrained=True).eval().to(device)
            cls_target = F.interpolate((target.to(device).to(torch.float32) / 127.5 - 1)[None], 224)
            logits = classifier(cls_target).softmax(1)
            classes = torch.multinomial(logits, w_avg_samples, replacement=True).squeeze()
            print(f'Main class: {logits.argmax(1).item()}, confidence: {logits.max().item():.4f}')
            c_samples = np.zeros([w_avg_samples, G.c_dim], dtype=np.float32)
            for i, c in enumerate(classes):
                c_samples[i, c] = 1
            c_samples = torch.from_numpy(c_samples).to(device)
        else:
            class_indices = torch.full((1,), class_idx).cuda()
            c = F.one_hot(class_indices, G.c_dim)
            c_samples = c.repeat([w_avg_samples, 1])


    # print(c_samples.shape)
    w_samples = G.mapping(z_samples, c_samples)  # [N, L, C]

    # get empirical w_avg
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    std = w_samples.std(0)

    # Load VGG16 feature detector.
    vgg16_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
    vgg16 = metric_utils.get_feature_detector(vgg16_url, device=device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    if centroids_path != "":
        with dnnlib.util.open_url(centroids_path, verbose=False) as f:
            w_centroids = np.load(f)
            center_w = torch.from_numpy(w_centroids['centers']).to(device)
        center_w = center_w.unsqueeze(1).repeat([1, G.num_ws, 1]).to(device)
        if center_idx != -1:
            print(f"Using centroids id :{center_idx}")
            w_avg = center_w[center_idx][0]
        else:
            center_images = G.synthesis(center_w)
            center_images = (center_images + 1) * (255/2)
            center_images  = F.interpolate(center_images, size=(target_images.shape[-2],target_images.shape[-1]) , mode='bicubic')

            center_features = vgg16(center_images, resize_images=False, return_lpips=True)
            lpips_dis = (target_features - center_features).square().sum(1)
            print(f"Using centroids id :{lpips_dis.argmin(0)}")
            w_avg = center_w[lpips_dis.argmin(0)][0]

    # initalize optimizer
    w_opt = torch.tensor(w_avg, dtype=torch.float32) # pylint: disable=not-callable
    std = torch.tensor(std, dtype=torch.float32).to(device)
    w_opt = w_opt.repeat(1,G.num_ws,1).to(device)
    w_offset = torch.zeros(w_opt.shape).to(device).requires_grad_(True)
    ema = ExponentialMovingAverage([w_offset], decay=0.995)
    optimizer = torch.optim.Adam([w_offset], betas=(0.9, 0.999), lr=initial_learning_rate)

    # run optimization loop
    all_images = []
    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        # for param_group in optimizer.opt.param_groups:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        synth_images = G.synthesis(w_opt+w_offset, noise_mode=noise_mode)

        # track images
        synth_images = (synth_images + 1) * (255/2)
        synth_images_np = synth_images.clone().detach().permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        all_images.append(synth_images_np)
        synth_images_ori = synth_images.clone()

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        gen_im_lr  = F.interpolate(synth_images_ori, size=(target_images.shape[-2],target_images.shape[-1]) , mode='bicubic')
        synth_features = vgg16(gen_im_lr, resize_images=False, return_lpips=True)
        lpips_loss = (target_features - synth_features).square().sum()
        l2_loss = l2_criterion(gen_im_lr, target_images) * 100
        reg = torch.norm(w_offset, 2)
        geocross = loss_geocross(w_offset, G.num_ws)

        rank_loss = 0
        if rank_w > 0:
            l_g_rank = netR(synth_images)
            l_g_rank = torch.sigmoid(l_g_rank-0.47)
            l_g_rank = torch.sum(l_g_rank)
            rank_loss = l_g_rank * rank_w

        # Step
        # optimizer.opt.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        loss = lpips_loss + l2_loss + rank_loss
        if offset_reg:
            loss += reg_w * reg
        loss += geocross * geo_reg
        # loss = l1_loss
        loss.backward()
        # optimizer.opt.step()
        optimizer.step()
        ema.update()
        msg  = f'[ step {step+1:>4d}/{num_steps}] '
        msg += f'[ lpips_loss: {float(lpips_loss):<5.2f}] '
        msg += f'[ l2_loss: {float(l2_loss):<5.4f}] '
        msg += f'[ reg: {float(reg):<5.2f}] '
        msg += f'[ geocross: {float(geocross):<5.2f}] '
        msg += f'[ rank: {float(rank_loss):<5.2f}] '
        msg += f'[ loss: {float(loss):<5.2f}] '
        if verbose: print(msg)

    if ema:
        ema.store()
    if c_samples is not None:
        return all_images, w_opt.detach(), (w_offset).detach(), c_samples.detach()
    else:
        return all_images, w_opt.detach(), (w_offset).detach(), None


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--seed', help='Random seed', type=int, default=42, show_default=True)
@click.option('--save-video', help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
@click.option('--inv-steps', help='Number of inversion steps', type=int, default=1000, show_default=True)
@click.option('--w-init', help='path to inital latent', type=str, default='', show_default=True)
@click.option('--run-pti', help='run pivotal tuning', is_flag=True)
@click.option('--pti-steps', help='Number of pti steps', type=int, default=350, show_default=True)
@click.option('--scale', help='lr scale', type=int, default=4, show_default=True)
@click.option('--offset-reg', help='regularize offset', is_flag=True)
@click.option('--class-idx', help='class-idx of the image', type=int, default=-1, show_default=True)
@click.option('--centroids_path', help='centroids', type=str, default="", show_default=True)
@click.option('--geo-reg', help='geocross', type=float, default=0,)
@click.option('--center-idx', help='center-idx of the image', type=int, default=-1, show_default=True)
@click.option('--bl2', help='using BL2 loss', is_flag=True)
@click.option('--rank-w', help='weight of rank loss', type=float, default=0,)
@click.option('--ema', help='using ema for optimize w', is_flag=True)
@click.option('--images-only', help='save images only', is_flag=True)
@click.option('--reg-w', help='weight of reg loss', type=float, default=0.5,)
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    inv_steps: int,
    w_init: str,
    run_pti: bool,
    pti_steps: int,
    scale: int,
    offset_reg: bool,
    class_idx: int,
    centroids_path: str,
    geo_reg: float,
    center_idx: int,
    bl2: bool,
    rank_w: float,
    ema: bool,
    images_only: bool,
    reg_w: float, 
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    D_kwargs = dnnlib.EasyDict(
        class_name='pg_modules.discriminator.ProjectedDiscriminator',
        backbones=['deit_base_distilled_patch16_224'],
        diffaug=True,
        interp224=False,
        backbone_kwargs=dnnlib.EasyDict(),
    )
    D_kwargs.backbone_kwargs.cout = 64
    D_kwargs.backbone_kwargs.expand = True
    D_kwargs.backbone_kwargs.proj_type = 2
    D_kwargs.backbone_kwargs.num_discs = 4
    D_kwargs.backbone_kwargs.cond = False
    common_kwargs = dict(c_dim=1000, img_resolution=512, img_channels=3)
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    with dnnlib.util.open_url(network_pkl) as fp:
        module = legacy.load_network_pkl(fp)
        G = module['G_ema'].to(device) # type: ignore

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    target_pil = target_pil.resize((G.img_resolution//scale, G.img_resolution//scale), PIL.Image.BICUBIC)
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil_up = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    target_up_uint8 = np.array(target_pil_up, dtype=np.uint8)

    # Latent optimization
    start_time = perf_counter()
    all_images = []
    if not w_init:
        print('Running Latent Optimization...')
        all_images, projected_w, offset_w, c = project(
            G, D, offset_reg, class_idx,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
            num_steps=inv_steps,
            device=device,
            verbose=True,
            noise_mode='const',
            centroids_path=centroids_path,
            geo_reg=geo_reg,
            center_idx=center_idx,
            bl2=bl2, rank_w=rank_w, ema=ema, reg_w=reg_w,
        )
        print(f'Elapsed time: {(perf_counter()-start_time):.1f} s')
    else:
        projected_w = torch.from_numpy(np.load(w_init)['w'])[0].to(device)

    start_time = perf_counter()

    # Run PTI
    if run_pti:
        print('Running Pivotal Tuning Inversion...')
        gen_images, G = pivotal_tuning(
            G, D, 
            projected_w, offset_w,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),
            device=device,
            num_steps=pti_steps,
            verbose=True, c=c,
            bl2=bl2, rank_w=0,
        )
        all_images += gen_images
        print(f'Elapsed time: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=60, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for synth_image in all_images:
            video.append_data(np.concatenate([target_up_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    PIL.Image.fromarray(target_up_uint8, 'RGB').save(f'{outdir}/target_up.png')
    synth_image = G.synthesis(projected_w+offset_w)
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')

    if not images_only:
        # save latents
        np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())
        np.savez(f'{outdir}/offset_w.npz', w=offset_w.unsqueeze(0).cpu().numpy())

        # Save Generator weights
        snapshot_data = {'G': G, 'G_ema': G}
        with open(f"{outdir}/G.pkl", 'wb') as f:
            dill.dump(snapshot_data, f)

    #----------------------------------------------------------------------------


if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

# optimize offsets for w+ space