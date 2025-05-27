import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import random
from basicsr.utils.registry import ARCH_REGISTRY
from scripts.utils import pad_tensor, pad_tensor_back


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas



def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3, stretch=False):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
        if stretch:
            from scripts.utils import stretch_linear
            betas = stretch_linear(betas)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


@ARCH_REGISTRY.register()
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        restore_fn=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.restore_fn = restore_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'],
            stretch=schedule_opt.get('stretch', False))
        
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    def norm_minus1_1(self, x):
        return (x - 0.5) * 2
    
    def norm_0_1(self, x):
        return (x + 1) * 0.5

    # use ddim to sample
    @torch.no_grad()
    def ddim_sample(
        self,
        x_in,
        ddim_timesteps=50,
        ddim_discr_method="uniform",
        ddim_eta=0.0,
        clip_denoised=True,
        continous=False,
        return_x_recon=False,
        return_pred_noise=False,
        return_all=False,
        pred_type='noise',
        clip_noise=False):

        if return_all:
            assert not (return_x_recon or return_pred_noise), "[return_x_recon, return_pred_noise, return_all], choose one or not!"
        assert not (return_x_recon and return_pred_noise), "[return_x_recon, return_pred_noise, return_all], choose one or not!"
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.num_timesteps // ddim_timesteps
            ddim_timestep_seq = list(reversed(range(self.num_timesteps - 1, -1, -c)))
            ddim_timestep_seq = np.asarray(ddim_timestep_seq)
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(self.num_timesteps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([-1]), ddim_timestep_seq[:-1])
        
        device = x_in.device
        b, c, h, w = x_in[:, :3, :, :].shape
        init_h = h
        init_w = w
        # start from pure noise (for each example in the batch)
        sample_img = torch.randn((b, c, init_h, init_w), device=device)
        sample_inter = (1 | (ddim_timesteps//10))
        ret_img = x_in[:, :3, :, :]
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            if return_all and i % sample_inter == 0:
                all_process = [F.interpolate(sample_img, (h, w))]
            t = torch.full((b,), ddim_timestep_seq[i], device=device, dtype=torch.long)

            noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(b, 1).to(device)

            prev_t = torch.full((b,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            if i == 0:
                alpha_cumprod_t_prev = torch.ones_like(alpha_cumprod_t)
            else:
                alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)

            if pred_type == 'noise':
                # 2. predict noise using model
                pred_noise = self.denoise_fn(torch.cat([F.interpolate(x_in, sample_img.shape[2:]), sample_img], dim=1), noise_level)
                if clip_noise:
                    pred_noise = torch.clamp(pred_noise, -1, 1)
                
                # 3. get the predicted x_0
                pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            else:
                assert False, "only pred noise"

            if return_all and i % sample_inter == 0:
                all_process.append(F.interpolate(pred_x0, (h, w)))
            
            pred_x0.clamp_(-1., 1.)
            
            if return_all and i % sample_inter == 0:
                all_process.append(F.interpolate(pred_x0, (h, w)))

            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            
            sample_already = False

            pred_x0, supervised_l_list, supervised_r_list, _ = self.restore_fn(self.norm_0_1(x_in), self.norm_0_1(pred_x0), noise_level)
            pred_x0.clamp_(0., 1.)

            pred_x0 = self.norm_minus1_1(pred_x0)
            
            if not sample_already:
                # compute variance: "sigma_t(η)" -> see DDIM formula (16)
                # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
                sigmas_t = ddim_eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
                
                # compute "direction pointing to x_t" of DDIM formula (12)
                pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
                
                # compute x_{t-1} of DDIM formula (12)
                x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(pred_x0)

                sample_img = x_prev
            
            if return_all and i % sample_inter == 0:
                all_process.append(F.interpolate(sample_img, (h, w)))

            if i % sample_inter == 0:
                if return_x_recon:
                    ret_img = torch.cat([ret_img, F.interpolate(pred_x0, (h, w))], dim=0)
                elif return_pred_noise:
                    ret_img = torch.cat([ret_img, pred_noise], dim=0)
                elif return_all:
                    ret_img = torch.cat([ret_img, torch.cat(all_process, dim=0)], dim=0)
                else:
                    ret_img = torch.cat([ret_img, F.interpolate(sample_img, (h, w))], dim=0)
        
        if continous:
            return ret_img, supervised_l_list, supervised_r_list
        else:
            return sample_img, supervised_l_list, supervised_r_list

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses_cs(self, x_HR, x_SR, noise=None, different_t_in_one_batch=False, clip_noise=False, t_range=None,
                    frozen_denoise=False):
        if not t_range:
            t_range = [1, self.num_timesteps]
        x_start = x_HR
        [b, c, h, w] = x_start.shape
        if different_t_in_one_batch:
            t = torch.randint(0, self.num_timesteps, (b,)).long() + 1
            t = t.to(x_start.device)
            continuous_sqrt_alpha_cumprod = self._extract(torch.from_numpy(self.sqrt_alphas_cumprod_prev), t, x_start.shape)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)
        else:
            t = np.random.randint(t_range[0], t_range[1] + 1)
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[t-1],
                    self.sqrt_alphas_cumprod_prev[t],
                    size=b
                )
            ).to(x_start.device)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
                b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)
        
        model_output = self.denoise_fn(
                    torch.cat([x_SR, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        if clip_noise:
            model_output = torch.clamp(model_output, -1, 1)

        self.noise = noise
        self.pred_noise = model_output
        self.x_recon = self.sqrt_recip_alphas_cumprod[t - 1] * x_noisy - self.sqrt_recipm1_alphas_cumprod[t - 1] * model_output
        self.x_recon = torch.clamp(self.x_recon, -1, 1)

        self.x_recon_detach = self.x_recon.detach()
        self.x_recon_output, supervised_l_list, supervised_r_list, l_MoE = self.restore_fn(self.norm_0_1(x_SR), self.norm_0_1(self.x_recon_detach),
                                                                                            continuous_sqrt_alpha_cumprod)
        self.x_recon_output = torch.clamp(self.x_recon_output, 0, 1)
        self.x_recon_output = self.norm_minus1_1(self.x_recon_output)
        
        return self.pred_noise, self.noise, self.x_recon_output, supervised_l_list, supervised_r_list, l_MoE

    def norm_0_1(self, x):
        return (x + 1) / 2.

    def norm_minus1_1(self, x):
        return (x * 2.) - 1.

    def forward(self, x_HR, x_SR, train_type='ddpm', *args, **kwargs):
        kwargs_cp = kwargs.copy()
        for k in kwargs_cp:
            if kwargs[k] is None:
                kwargs.pop(k)
        return self.p_losses_cs(x_HR, x_SR, *args, **kwargs)
