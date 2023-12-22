import os
import copy
import math
import argparse
import numpy as np
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from time import time
from tqdm import tqdm
from easydict import EasyDict

import torch
import torchvision
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from data import get_metadata, get_dataset, fix_legacy_dict
import unets

unsqueeze3x = lambda x: x[..., None, None, None]

# Define the wavenumbers
kx = 2 * np.pi * np.fft.fftfreq(64, d=1/63)
ky = 2 * np.pi * np.fft.fftfreq(64, d=1/63)

kx,ky = np.meshgrid(kx, ky)

def laplacian_spectral_bc(u,sigma=1):
    # Compute the 2D Fourier transform of u
    u_hat = np.fft.fft2(u[:,:])

    # Compute Laplacian in Fourier space
    laplacian_u_hat = -(kx**2 + ky**2) * u_hat
    laplacian_u = np.real(np.fft.ifft2(laplacian_u_hat))

    return np.clip(laplacian_u,-1,1)

class GuassianDiffusion:
    """Gaussian diffusion process with 1) Cosine schedule for beta values (https://arxiv.org/abs/2102.09672)
    2) L_simple training objective from https://arxiv.org/abs/2006.11239.
    """

    def __init__(self, timesteps=1000, device="cuda:0"):
        self.timesteps = timesteps
        self.device = device
        self.alpha_bar_scheduler = (
            lambda t: math.cos((t / self.timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        self.scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, self.timesteps, self.device
        )

        self.clamp_x0 = lambda x: x.clamp(-1, 1)
        self.get_x0_from_xt_eps = lambda xt, eps, t, scalars: (
            self.clamp_x0(
                1
                / unsqueeze3x(scalars.alpha_bar[t].sqrt())
                * (xt - unsqueeze3x((1 - scalars.alpha_bar[t]).sqrt()) * eps)
            )
        )
        self.get_pred_mean_from_x0_xt = (
            lambda xt, x0, t, scalars: unsqueeze3x(
                (scalars.alpha_bar[t].sqrt() * scalars.beta[t])
                / ((1 - scalars.alpha_bar[t]) * np.maximum(scalars.alpha[t],1e-5).sqrt())
            )
            * x0
            + unsqueeze3x(
                (scalars.alpha[t] - scalars.alpha_bar[t])
                / ((1 - scalars.alpha_bar[t]) * np.maximum(scalars.alpha[t],1e-5).sqrt())
            )
            * xt
        )

    def get_all_scalars(self, alpha_bar_scheduler, timesteps, device, betas=None):
        """
        Using alpha_bar_scheduler, get values of all scalars, such as beta, beta_hat, alpha, alpha_hat, etc.
        """
        all_scalars = {}
        if betas is None:
            all_scalars["beta"] = torch.from_numpy(
                np.array(
                    [
                        min(
                            1 - alpha_bar_scheduler(t + 1) / alpha_bar_scheduler(t),
                            0.999,
                        )
                        for t in range(timesteps)
                    ]
                )
            ).to(
                device
            )  # hardcoding beta_max to 0.999
        else:
            all_scalars["beta"] = betas
        all_scalars["beta_log"] = torch.log(all_scalars["beta"])
        all_scalars["alpha"] = 1 - all_scalars["beta"]
        all_scalars["alpha_bar"] = torch.cumprod(all_scalars["alpha"], dim=0)
        all_scalars["beta_tilde"] = (
            all_scalars["beta"][1:]
            * (1 - all_scalars["alpha_bar"][:-1])
            / (1 - all_scalars["alpha_bar"][1:])
        )
        all_scalars["beta_tilde"] = torch.cat(
            [all_scalars["beta_tilde"][0:1], all_scalars["beta_tilde"]]
        )
        all_scalars["beta_tilde_log"] = torch.log(all_scalars["beta_tilde"])
        return EasyDict(dict([(k, v.float()) for (k, v) in all_scalars.items()]))

    def sample_from_forward_process(self, x0, t):
        """Single step of the forward process, where we add noise in the image.
        Note that we will use this paritcular realization of noise vector (eps) in training.
        """
        eps = torch.randn_like(x0)
        xt = (
            unsqueeze3x(self.scalars.alpha_bar[t].sqrt()) * x0
            + unsqueeze3x((1 - self.scalars.alpha_bar[t]).sqrt()) * eps
        )
        return xt.float(), eps

    def sample_from_reverse_process(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False
    ):
        """Sampling images by iterating over all timesteps.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better image quality.

        Return: An image tensor with identical shape as XT.
        """
        model.eval()
        final = xT
        u = xT[:,0,:,:]

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, 4*timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            print(t)
            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)
                pred_epsilon = model(final, current_t, **model_kwargs)
                # using xt+x0 to derive mu_t, instead of using xt+eps (former is more stable)
                pred_x0 = self.get_x0_from_xt_eps(
                    final, pred_epsilon, current_sub_t, scalars
                )
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        final = (
                            unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1]).sqrt()
                            * pred_x0
                            + (
                                1 - unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1])
                            ).sqrt()
                            * pred_epsilon
                        )
                    else:
                        final = pred_mean + unsqueeze3x(
                            scalars.beta_tilde[current_sub_t].sqrt()
                        ) * torch.randn_like(final)
                final = final.detach()
                final[:,0,:,:] = u
        return final

    def sample_from_restoration_process(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False
    ):
        """Sampling images by iterating over all timesteps.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better image quality.

        Return: An image tensor with identical shape as XT.
        """
        model.eval()
        final = xT
        u = xT[:,0,:,:]

        # Restoration arguments
        sigma_f = model_kwargs["sigma_y"]
        eta = model_kwargs["eta"]
        eta_b = model_kwargs["eta_b"]

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, 4*timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            print(t)
            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)

                x_theta = self.sample_from_reverse_process(
                    model, final, timesteps=timesteps-t, model_kwargs=model_kwargs, ddim=ddim
                )

                new_kx = kx.copy()
                new_ky = ky.copy()

                sigma_t = np.sqrt(scalars["alpha_bar"][current_sub_t - 1])
                k_index = sigma_t < sigma_f*(kx**2+ky**2)
                not_k_index = sigma_t >= sigma_f*(kx**2+ky**2)
                
                for n in range(x_theta.shape[0]):
                    u_i = x_theta[n,0,:,:].numpy()
                    u_bar_i = np.fft.fft2(u_bar_i)
                    f_theta_i = x_theta[n,1,:,:].numpy()
                    f_theta_i_bar = np.fft.fft2(f_theta_i)

                    f_theta_i_bar[k_index] = f_theta_i_bar[k_index] + np.sqrt(1-eta**2)*sigma_t*(kx[k_index]**2+ky[k_index]**2)*(u_bar_i[k_index]-f_theta_i_bar[k_index])
                    f_theta_i_bar[not_k_index] = (1-eta_b)*f_theta_i_bar[not_k_index] + eta_b*u_bar_i[not_k_index]

                pred_epsilon = model(final, current_t, **model_kwargs)
                # using xt+x0 to derive mu_t, instead of using xt+eps (former is more stable)
                pred_x0 = self.get_x0_from_xt_eps(
                    final, pred_epsilon, current_sub_t, scalars
                )
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        final = (
                            unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1]).sqrt()
                            * pred_x0
                            + (
                                1 - unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1])
                            ).sqrt()
                            * pred_epsilon
                        )
                    else:
                        final = pred_mean + unsqueeze3x(
                            scalars.beta_tilde[current_sub_t].sqrt()
                        ) * torch.randn_like(final)
                final = final.detach()
                final[:,0,:,:] = u
        return final


class loss_logger:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.loss = []
        self.start_time = time()
        self.ema_loss = None
        self.ema_w = 0.9

    def log(self, v, display=False):
        self.loss.append(v)
        if self.ema_loss is None:
            self.ema_loss = v
        else:
            self.ema_loss = self.ema_w * self.ema_loss + (1 - self.ema_w) * v

        if display:
            print(
                f"Steps: {len(self.loss)}/{self.max_steps} \t loss (ema): {self.ema_loss:.3f} "
                + f"\t Time elapsed: {(time() - self.start_time)/3600:.3f} hr"
            )


def train_one_epoch(
    model,
    dataloader,
    diffusion,
    optimizer,
    logger,
    lrs,
    args,
):
    model.train()
    for step, images in enumerate(dataloader):
        # must use [-1, 1] pixel range for images
        if args.dataset == "poisson":
            images, labels = (
                images.to(args.device),
                labels.to(args.device) if args.class_cond else None,
            )
        else:
            images, labels = images
            assert (images.max().item() <= 1) and (0 <= images.min().item())
            images, labels = (
                2 * images.to(args.device) - 1,
                labels.to(args.device) if args.class_cond else None,
            )
        t = torch.randint(diffusion.timesteps, (len(images),), dtype=torch.int64).to(
            args.device
        )
        xt, eps = diffusion.sample_from_forward_process(images, t)
        pred_eps = model(xt, t, y=labels)

        loss = ((pred_eps - eps) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lrs is not None:
            lrs.step()

        # update ema_dict
        if args.local_rank == 0:
            new_dict = model.state_dict()
            for (k, v) in args.ema_dict.items():
                args.ema_dict[k] = (
                    args.ema_w * args.ema_dict[k] + (1 - args.ema_w) * new_dict[k]
                )
            logger.log(loss.item(), display=not step % 100)


def sample_N_images(
    N,
    model,
    diffusion,
    xT=None,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    image_size=32,
    num_classes=None,
    args=None,
):
    """use this function to sample any number of images from a given
        diffusion model and diffusion process.

    Args:
        N : Number of images
        model : Diffusion model
        diffusion : Diffusion process
        xT : Starting instantiation of noise vector.
        sampling_steps : Number of sampling steps.
        batch_size : Batch-size for sampling.
        num_channels : Number of channels in the image.
        image_size : Image size (assuming square images).
        num_classes : Number of classes in the dataset (needed for class-conditioned models)
        args : All args from the argparser.

    Returns: Numpy array with N images and corresponding labels.
    """
    samples, labels, num_samples = [], [], 0
    N = 64
    data_list = []
    for i in range(10001,10065):
        xT_i = torch.load("dataset/Poisson/seed_"+str(i)+".pt")
        # Dry sampling of f_T
        xT_i[1] = 0.0
        data_list.append(xT_i)
    xT = torch.stack(data_list)
    y = None
    gen_images = diffusion.sample_from_reverse_process(
        model, xT, sampling_steps, {"y": y}, args.ddim
    )
    return (gen_images, None)

def restore_N_images(
    N,
    model,
    diffusion,
    xT=None,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    image_size=32,
    num_classes=None,
    args=None,
):
    """use this function to sample any number of images from a given
        diffusion model and diffusion process.

    Args:
        N : Number of images
        model : Diffusion model
        diffusion : Diffusion process
        xT : Starting instantiation of noise vector.
        sampling_steps : Number of sampling steps.
        batch_size : Batch-size for sampling.
        num_channels : Number of channels in the image.
        image_size : Image size (assuming square images).
        num_classes : Number of classes in the dataset (needed for class-conditioned models)
        args : All args from the argparser.

    Returns: Numpy array with N images and corresponding labels.
    """
    samples, labels, num_samples = [], [], 0
    N = 64
    data_list = []
    for i in range(10001,10065):
        xT_i = torch.load("dataset/Poisson/seed_"+str(i)+".pt")
        # Sampling of f_T with DDRM
        xT_i[1] = torch.FloatTensor(laplacian_spectral_bc(xT_i[0]))
        data_list.append(xT_i)
    xT = torch.stack(data_list)
    y = None
    for i in range(sampling_steps):
        gen_images = diffusion.sample_from_restoration_process(
            model, xT, sampling_steps, {"y": y, "sigma_y": args.sigma_y, "eta": args.eta, "eta_b": args.eta_b}, args.ddim
        )
    return (gen_images, None)

def main():
    parser = argparse.ArgumentParser("Minimal implementation of diffusion models")
    # diffusion model
    parser.add_argument("--arch", type=str, help="Neural network architecture", default="UNet")
    parser.add_argument(
        "--class-cond",
        action="store_true",
        default=False,
        help="train class-conditioned diffusion model",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=1000,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=250,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        default=False,
        help="Sampling using DDIM update step",
    )
    # dataset
    parser.add_argument("--dataset", type=str, default="poisson")
    parser.add_argument("--data-dir", type=str, default="./dataset/")
    # optimizer
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch-size per gpu"
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--ema_w", type=float, default=0.9995)
    # sampling/finetuning
    parser.add_argument("--pretrained-ckpt", type=str, help="Pretrained model ckpt", default="./trained_models/UNet_poisson-epoch_500-timesteps_1000-class_condn_False.pt")
    parser.add_argument("--delete-keys", nargs="+", help="Pretrained model ckpt")
    parser.add_argument(
        "--sampling-only",
        action="store_true",
        default=False,
        help="No training, just sample images (will save them in --save-dir)",
    )
    parser.add_argument(
        "--restoration-only",
        action="store_true",
        default=True,
        help="No training, just restore images (will save them in --save-dir)",
    )
    # Restoration arguments
    parser.add_argument("--sigma_y", default=float(1e-5), type=float)
    parser.add_argument("--eta_b", default=0.9, type=float)
    parser.add_argument("--eta", default=0.8, type=float)

    parser.add_argument(
        "--num-sampled-images",
        type=int,
        default=50000,
        help="Number of images required to sample from the model",
    )

    # misc
    parser.add_argument("--save-dir", type=str, default="./trained_models/")
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--seed", default=112233, type=int)

    # setup
    args = parser.parse_args()
    metadata = get_metadata(args.dataset)
    os.makedirs(args.save_dir, exist_ok=True)
    # torch.backends.cudnn.benchmark = True
    args.device = "cpu"
    # torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    if args.local_rank == 0:
        print(args)

    # Creat model and diffusion process
    model = unets.__dict__[args.arch](
        image_size=metadata.image_size,
        in_channels=metadata.num_channels,
        out_channels=metadata.num_channels,
        num_classes=metadata.num_classes if args.class_cond else None,
    ).to(args.device)
    if args.local_rank == 0:
        print(
            "We are assuming that model input/ouput pixel range is [-1, 1]. Please adhere to it."
        )
    diffusion = GuassianDiffusion(args.diffusion_steps, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # load pre-trained model
    if args.pretrained_ckpt:
        print(f"Loading pretrained model from {args.pretrained_ckpt}")
        d = fix_legacy_dict(torch.load(args.pretrained_ckpt, map_location=args.device))
        dm = model.state_dict()
        if args.delete_keys:
            for k in args.delete_keys:
                print(
                    f"Deleting key {k} becuase its shape in ckpt ({d[k].shape}) doesn't match "
                    + f"with shape in model ({dm[k].shape})"
                )
                del d[k]
        model.load_state_dict(d, strict=False)
        print(
            f"Mismatched keys in ckpt and model: ",
            set(d.keys()) ^ set(dm.keys()),
        )
        print(f"Loaded pretrained model from {args.pretrained_ckpt}")

    # distributed training
    ngpus = 0
    if ngpus > 1:
        if args.local_rank == 0:
            print(f"Using distributed training on {ngpus} gpus.")
        args.batch_size = args.batch_size // ngpus
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # sampling
    if args.sampling_only:
        sampled_images, labels = sample_N_images(
            args.num_sampled_images,
            model,
            diffusion,
            None,
            args.sampling_steps,
            args.batch_size,
            metadata.num_channels,
            metadata.image_size,
            metadata.num_classes,
            args,
        )
        torch.save(sampled_images,
                    os.path.join(
                        args.save_dir,
                        f"{args.arch}_{args.dataset}-{args.diffusion_steps}_steps-{args.sampling_steps}-sampling_steps-class_condn_{args.class_cond}.pt",
                    ))
        return

    # restoration
    if args.restoring_only:
        sampled_images, labels = restore_N_images(
            args.num_sampled_images,
            model,
            diffusion,
            None,
            args.sampling_steps,
            args.batch_size,
            metadata.num_channels,
            metadata.image_size,
            metadata.num_classes,
            args,
        )
        torch.save(sampled_images,
                    os.path.join(
                        args.save_dir,
                        f"{args.arch}_{args.dataset}-{args.diffusion_steps}_steps-{args.sampling_steps}-sampling_steps-class_condn_{args.class_cond}.pt",
                    ))
        return

    # Load dataset
    train_set = get_dataset(args.dataset, args.data_dir, metadata)
    sampler = DistributedSampler(train_set) if ngpus > 1 else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    if args.local_rank == 0:
        print(
            f"Training dataset loaded: Number of batches: {len(train_loader)}, Number of images: {len(train_set)}"
        )
    logger = loss_logger(len(train_loader) * args.epochs)

    # ema model
    args.ema_dict = copy.deepcopy(model.state_dict())

    # lets start training the model
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, diffusion, optimizer, logger, None, args)
        if args.local_rank == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.save_dir,
                    f"{args.arch}_{args.dataset}-epoch_{args.epochs}-timesteps_{args.diffusion_steps}-class_condn_{args.class_cond}.pt",
                ),
            )
            torch.save(
                args.ema_dict,
                os.path.join(
                    args.save_dir,
                    f"{args.arch}_{args.dataset}-epoch_{args.epochs}-timesteps_{args.diffusion_steps}-class_condn_{args.class_cond}_ema_{args.ema_w}.pt",
                ),
            )
        if not epoch % 1:
            sampled_images, _ = sample_N_images(
                64,
                model,
                diffusion,
                None,
                args.sampling_steps,
                args.batch_size,
                metadata.num_channels,
                metadata.image_size,
                metadata.num_classes,
                args,
            )
            if args.local_rank == 0:
                torch.save(sampled_images,
                            os.path.join(
                                args.save_dir,
                                f"{args.arch}_{args.dataset}-{args.diffusion_steps}_steps-{args.sampling_steps}-sampling_steps-class_condn_{args.class_cond}.pt",
                            ))
if __name__ == "__main__":
    main()
