import glob
import os
from functools import partial
from itertools import chain

import hydra
import lpips as lpips_lib
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from ema_pytorch import EMA
from lightning.fabric import Fabric
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_accuracy

import wandb
from datasets.dataset_factory import get_dataset
from datasets.shared_dataset import MaskedDataset
from eval import evaluate_dataset
from gan.discriminator import Discriminator
from gan.get3d.discriminator import Discriminator as GET3DDiscriminator
from gaussian_renderer import render_predicted
from scene.gaussian_predictor import GaussianSplatPredictor
from utils.general_utils import (
    adjust_channels,
    collate_and_superimpose,
    mask_to_outline,
    occluded_area,
    safe_state,
)
from utils.loss_utils import l1_loss, l2_loss


def compute_gan_loss(
    cfg, d_net: nn.Module, criterion, fake, real, in_view, camera_pose, b_idxes
):
    RGB = 3
    CH = 1  # Channel dimension
    label_fake = torch.zeros
    label_real = torch.ones
    n = fake.shape[0]
    sub_n = max(int(n * cfg.gan.sample_p), 1)

    assert cfg.gan.enabled
    assert fake.shape == real.shape == input.view.shape

    def background_pixels(x):
        # This is not exactly perfect
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        return r == g == b == (1.0 if cfg.data.white_background else 0.0)

    assert in_view.shape[1] == 4
    fake_idx, real_idx = torch.randperm(n)[:sub_n], torch.randperm(n)[:sub_n]
    if not cfg.gan.discriminator_occl_mask:
        in_view = in_view[:, :RGB, ...]
        # Add input view as another channel
        fake = torch.cat(fake[idx], in_view[fake_idx], dim=CH)
        real = torch.cat(real[idx], in_view[real_idx], dim=CH)
    # Add mask channel
    fake = torch.cat(background_pixels(fake[:, :RGB]), dim=CH)
    real = torch.cat(background_pixels(real[:, :RGB]), dim=CH)
    # The channel dimension now is composed of RGB + RGB of input view* + mask of RGB
    # *depends on cfg
    fake_c, real_c = camera_pose[fake_idx], camera_pose[real_idx]

    # Discriminator
    y, y_mask = d_net(img=torch.cat([fake, real]), c=torch.cat[fake_c, real_c])[:, 0]
    y_hat = torch.cat([label_fake(sub_n), label_real(sub_n)]).to(fake.device)
    d_loss = criterion(y, y_hat)
    d_loss_mask = criterion(y_mask, y_hat)
    # Generator, note that labels are swapped
    g_y, g_y_mask = netD(img=fake, c=fake_c)
    g_y_hat = label_real(sub_n).to(fake.device)
    g_loss = criterion(g_y, g_y_hat)
    g_loss_mask = criterion(g_y_mask, g_y_hat)

    with torch.no_grad():
        loss_dict = {
            "d_loss_rgb_fake": d_loss[:sub_n].mean().item(),
            "d_loss_rgb_real": d_loss[sub_n:].mean().item(),
            "d_loss_mask_fake": d_loss_mask[:sub_n].mean().item(),
            "d_loss_mask_real": d_loss_mask[sub_n:].mean().item(),
            "d_acc_rgb": binary_accuracy(y, y_hat),
            "d_acc_mask": binary_accuracy(y_mask, y_hat),
        }
    return {
        **loss_dict,
        "d_loss_rgb": d_loss.mean(),
        "d_loss_mask": d_loss_mask.mean(),
        "g_loss_rgb": g_loss.mean(),
        "g_loss_mask": g_loss_mask.mean(),
    }


@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def main(cfg: DictConfig):

    torch.set_float32_matmul_precision("high")
    if cfg.general.mixed_precision:
        fabric = Fabric(
            accelerator="cuda",
            devices=cfg.general.num_devices,
            strategy="ddp",
            precision="16-mixed",
        )
    else:
        fabric = Fabric(
            accelerator="cuda", devices=cfg.general.num_devices, strategy="ddp"
        )
    fabric.launch()

    if fabric.is_global_zero:
        vis_dir = os.getcwd()

        dict_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        if os.path.isdir(os.path.join(vis_dir, "wandb")):
            run_name_path = glob.glob(
                os.path.join(vis_dir, "wandb", "latest-run", "run-*")
            )[0]
            print("Got run name path {}".format(run_name_path))
            run_id = os.path.basename(run_name_path).split("run-")[1].split(".wandb")[0]
            print("Resuming run with id {}".format(run_id))
            wandb_run = wandb.init(
                project=cfg.wandb.project, resume=True, id=run_id, config=dict_cfg
            )

        else:
            wandb_run = wandb.init(
                project=cfg.wandb.project, reinit=True, config=dict_cfg
            )

    first_iter = 0
    device = safe_state(cfg)

    gaussian_predictor = GaussianSplatPredictor(cfg)
    gaussian_predictor = gaussian_predictor.to(memory_format=torch.channels_last)

    l = []
    if cfg.model.network_with_offset:
        l.append(
            {
                "params": gaussian_predictor.network_with_offset.parameters(),
                "lr": cfg.opt.base_lr,
            }
        )
    if cfg.model.network_without_offset:
        l.append(
            {
                "params": gaussian_predictor.network_wo_offset.parameters(),
                "lr": cfg.opt.base_lr,
            }
        )
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, betas=cfg.opt.betas)

    if cfg.gan.enabled == True:
        assert cfg.opt.lambda_g_gan > 0.0
        # Input the occluded input view as additional channels to
        discriminator = GET3DDiscriminator(
            c_dim=2, img_resolution=cfg.data.training_resolution, img_channels=3 * 2
        ).to(
            "cuda"
        )  # Discriminator("tmp").to("cuda")
        d_optimizer = torch.optim.Adam(
            {
                "params": discriminator.parameters(),
                "lr": cfg.opt.gan.base_lr,
            },
            lr=0.0,
            eps=1e-15,
            betas=cfg.opt.betas,
        )

    # Resuming training
    if fabric.is_global_zero:
        if os.path.isfile(os.path.join(vis_dir, "model_latest.pth")):
            print(
                "Loading an existing model from ",
                os.path.join(vis_dir, "model_latest.pth"),
            )
            checkpoint = torch.load(
                os.path.join(vis_dir, "model_latest.pth"), map_location=device
            )
            try:
                gaussian_predictor.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError:
                gaussian_predictor.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )
                print("Warning, model mismatch - was this expected?")
            first_iter = checkpoint["iteration"]
            best_PSNR = checkpoint["best_PSNR"]
            print("Loaded model")
        # Resuming from checkpoint
        elif cfg.opt.pretrained_ckpt is not None:
            pretrained_ckpt_dir = os.path.join(
                cfg.opt.pretrained_ckpt, "model_latest.pth"
            )
            checkpoint = torch.load(pretrained_ckpt_dir, map_location=device)
            try:
                gaussian_predictor.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError:
                gaussian_predictor.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )
            best_PSNR = checkpoint["best_PSNR"]
            del checkpoint  # Unload duplicate checkpoint data
            print("Loaded model from a pretrained checkpoint")
        else:
            best_PSNR = 0.0

    if cfg.opt.freeze_decoder:
        # Freeze the decoder. The decoder seems to be split into two
        print("Freezing decoder. Only encoder will be fine-tuned")
        for param in chain(
            gaussian_predictor.network_with_offset.encoder.dec.parameters(),
            gaussian_predictor.network_with_offset.out.parameters(),
        ):
            param.requires_grad = False

    if cfg.opt.ema.use and fabric.is_global_zero:
        ema = EMA(
            gaussian_predictor,
            beta=cfg.opt.ema.beta,
            update_every=cfg.opt.ema.update_every,
            update_after_step=cfg.opt.ema.update_after_step,
        )
        ema = fabric.to_device(ema)

    if cfg.opt.loss == "l2":
        loss_fn = l2_loss
    elif cfg.opt.loss == "l1":
        loss_fn = l1_loss

    if cfg.opt.lambda_lpips != 0:
        lpips_fn = fabric.to_device(lpips_lib.LPIPS(net="vgg"))
    lambda_lpips = cfg.opt.lambda_lpips
    lambda_l12 = 1.0 - lambda_lpips
    lambda_g_gan = cfg.opt.lambda_g_gan
    bce = nn.BCEWithLogitsLoss(reduction="none")

    bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32)
    background = fabric.to_device(background)

    if cfg.data.category in ["nmr", "objaverse"]:
        num_workers = 2
        persistent_workers = True
    else:
        num_workers = 0
        persistent_workers = False

    dataset = MaskedDataset(cfg, get_dataset(cfg, "train"))
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.opt.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=2,
        collate_fn=partial(collate_and_superimpose, cfg.data.input_images, 200),
    )

    val_dataset = MaskedDataset(
        cfg, get_dataset(cfg, "val"), return_superimposed_input=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )

    test_dataset = MaskedDataset(
        cfg, get_dataset(cfg, "vis"), return_superimposed_input=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # distribute model and training dataset
    gaussian_predictor, optimizer = fabric.setup(gaussian_predictor, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)

    gaussian_predictor.train()
    print(f"Training dataset size = {len(dataloader)}")

    print("Beginning training")
    first_iter += 1
    iteration = first_iter

    for num_epoch in range(
        (cfg.opt.iterations + 1 - first_iter) // len(dataloader) + 1
    ):
        dataloader.sampler.set_epoch(num_epoch)

        for data, overlay_data, input_data in dataloader:
            iteration += 1

            print(
                "starting iteration {} -> {} on process {}".format(
                    iteration, iteration / cfg.opt.iterations, fabric.global_rank
                )
            )

            # =============== Prepare input ================
            rot_transform_quats = data["source_cv2wT_quat"][:, : cfg.data.input_images]

            with torch.no_grad():
                if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                    assert False
                else:
                    focals_pixels_pred = None
                    input_images = input_data  # torch.cat(
                    #    [
                    #        transforms.ColorJitter(brightness=0.0, hue=0.0)(
                    #            input_data[:, :, :3]
                    #        ),
                    #        input_data[:, :, 3:4],
                    #    ],
                    #    dim=2,
                    # )
                    input_images = adjust_channels(cfg, input_images)

            gaussian_splats = gaussian_predictor(
                input_images,
                data["view_to_world_transforms"][:, : cfg.data.input_images, ...],
                rot_transform_quats,
                focals_pixels_pred,
            )

            if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                assert False
            # Render
            l12_loss_sum = 0.0
            lpips_loss_sum = 0.0
            rendered_images = []
            gt_images = []
            loss_weight = []
            camera_pose = []
            b_idxes = []

            for b_idx in range(data["gt_images"].shape[0]):
                # image at index 0 is training, remaining images are targets
                # Rendering is done sequentially because gaussian rasterization code
                # does not support batching
                gaussian_splat_batch = {
                    k: v[b_idx].contiguous() for k, v in gaussian_splats.items()
                }
                for r_idx in range(
                    0 if cfg.opt.compute_loss_on_condition else cfg.data.input_images,
                    data["gt_images"].shape[1],
                ):
                    if "focals_pixels" in data.keys():
                        focals_pixels_render = data["focals_pixels"][b_idx, r_idx].cpu()
                    else:
                        focals_pixels_render = None
                    image = render_predicted(
                        gaussian_splat_batch,
                        data["world_view_transforms"][b_idx, r_idx],
                        data["full_proj_transforms"][b_idx, r_idx],
                        data["camera_centers"][b_idx, r_idx],
                        background,
                        cfg,
                        focals_pixels=focals_pixels_render,
                    )["render"]
                    if cfg.opt.weight_loss.enabled:
                        n_weights = 0
                        lw = torch.zeros_like(image)
                        if (
                            r_idx < cfg.data.input_images
                            and cfg.opt.weight_loss.occluded_area > 0.0
                        ):
                            # WARN: Input image is currently always skipped
                            lw += (
                                occluded_area(
                                    data["gt_images"][b_idx, r_idx, 3:4],
                                    overlay_data["gt_images"][b_idx, r_idx, 3:4],
                                )
                                * cfg.opt.weight_loss.occluded_area
                            )
                            n_weights += 1
                        if cfg.opt.weight_loss.outline:
                            lw += (
                                mask_to_outline(input_images[b_idx, 0, 3:4])
                                * cfg.opt.weight_loss.outline
                            )
                            n_weights += 1
                        if n_weights > 0:
                            lw = (
                                cfg.opt.weight_loss.offset
                                + lw / n_weights * cfg.opt.weight_loss.global_coef
                            )
                        else:
                            lw = torch.ones_like(lw)
                        loss_weight.append(lw)
                    # Put in a list for a later loss computation
                    rendered_images.append(image)
                    gt_images.append(data["gt_images"][b_idx, r_idx, :3])
                    camera_pose.append(data["world_view_transforms"][b_idx, r_idx])
                    b_idxes.append(b_idx)
            rendered_images = torch.stack(rendered_images, dim=0)
            gt_images = torch.stack(gt_images, dim=0)
            print("camera pose1", camera_pose[0].shape)
            camera_pose = torch.stack(camera_pose, dim=0)
            print("camera pose2", camera_pose.shape)
            b_idxes = torch.LongTensor(b_idxes)
            loss_weight = (
                torch.stack(loss_weight, dim=0) if cfg.opt.weight_loss.enabled else None
            )

            if cfg.gan.enabled == True:
                gan_loss = compute_gan_loss(
                    cfg,
                    discriminator,
                    bce,
                    rendered_images,
                    gt_images,
                    camera_pose,
                    b_idxes,
                )
                d_gan_loss = (gan_loss["d_loss_rgb"] + gan_loss["d_loss_mask"]) / 2.0
                g_gan_loss = (gan_loss["g_loss_rgb"] + gan_loss["g_loss_mask"]) / 2.0
                wandb.log(
                    {"d_gan_loss": d_gan_loss, "g_gan_loss": g_gan_loss, **gan_metrics},
                    step=iteration,
                )
            else:
                gan_loss = 0.0

            # Loss computation
            l12_loss_sum = loss_fn(rendered_images, gt_images, loss_weight)
            if cfg.opt.lambda_lpips != 0:
                lpips_loss_sum = torch.mean(
                    lpips_fn(rendered_images * 2 - 1, gt_images * 2 - 1),
                )

            total_loss = (
                l12_loss_sum * lambda_l12
                + lpips_loss_sum * lambda_lpips
                + g_gan_loss * lambda_g_gan
            )
            if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                assert False

            assert not total_loss.isnan(), "Found NaN loss!"
            print(
                "finished forward {} ({}) on process {}".format(
                    iteration, iteration / cfg.opt.iterations, fabric.global_rank
                )
            )
            fabric.backward(total_loss)
            d_gan_loss.backward()

            # ============ Optimization ===============
            optimizer.step()
            optimizer.zero_grad()
            d_optimizer.step()
            d_optimizer.zero_grad()
            print(
                "finished opt {} ({}) on process {}".format(
                    iteration, iteration / cfg.opt.iterations, fabric.global_rank
                )
            )

            if cfg.opt.ema.use and fabric.is_global_zero:
                ema.update()

            print(
                "finished iteration {} ({}) on process {}".format(
                    iteration, iteration / cfg.opt.iterations, fabric.global_rank
                )
            )

            gaussian_predictor.eval()

            # ========= Logging =============
            with torch.no_grad():
                if iteration % cfg.logging.loss_log == 0 and fabric.is_global_zero:
                    wandb.log(
                        {"training_loss": np.log10(total_loss.item() + 1e-8)},
                        step=iteration,
                    )
                    if cfg.opt.lambda_lpips != 0:
                        wandb.log(
                            {"training_l12_loss": np.log10(l12_loss_sum.item() + 1e-8)},
                            step=iteration,
                        )
                        wandb.log(
                            {
                                "training_lpips_loss": np.log10(
                                    lpips_loss_sum.item() + 1e-8
                                )
                            },
                            step=iteration,
                        )
                    if (
                        cfg.data.category == "hydrants"
                        or cfg.data.category == "teddybears"
                    ):
                        if type(big_gaussian_reg_loss) == float:
                            brl_for_log = big_gaussian_reg_loss
                        else:
                            brl_for_log = big_gaussian_reg_loss.item()
                        if type(small_gaussian_reg_loss) == float:
                            srl_for_log = small_gaussian_reg_loss
                        else:
                            srl_for_log = small_gaussian_reg_loss.item()
                        wandb.log(
                            {"reg_loss_big": np.log10(brl_for_log + 1e-8)},
                            step=iteration,
                        )
                        wandb.log(
                            {"reg_loss_small": np.log10(srl_for_log + 1e-8)},
                            step=iteration,
                        )

                if (
                    iteration % cfg.logging.render_log == 0 or iteration == 1
                ) and fabric.is_global_zero:
                    b_idx = 0
                    r_idx = (
                        0
                        if cfg.opt.compute_loss_on_condition
                        else cfg.data.input_images
                    )
                    wandb.log(
                        {
                            "render": wandb.Image(
                                rendered_images[b_idx, r_idx, :3, ...]
                                .clamp(0.0, 1.0)
                                .permute(1, 2, 0)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        },
                        step=iteration,
                    )
                    wandb.log(
                        {
                            "occluded_input": wandb.Image(
                                input_images[b_idx, 0, :3, ...]
                                .permute(1, 2, 0)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        },
                        step=iteration,
                    )
                    wandb.log(
                        {
                            "mask": wandb.Image(
                                input_images[b_idx, 0, 3:4, ...]
                                .permute(1, 2, 0)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        },
                        step=iteration,
                    )
                    wandb.log(
                        {
                            "gt": wandb.Image(
                                data["gt_images"][b_idx, r_idx, :3]
                                .permute(1, 2, 0)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        },
                        step=iteration,
                    )
                if (
                    iteration % cfg.logging.loop_log == 0 or iteration == 1
                ) and fabric.is_global_zero:
                    # torch.cuda.empty_cache()
                    try:
                        vis_data, _, vis_input_data = next(test_iterator)
                    except UnboundLocalError:
                        test_iterator = iter(test_dataloader)
                        vis_data, _, vis_input_data = next(test_iterator)
                    except StopIteration or UnboundLocalError:
                        test_iterator = iter(test_dataloader)
                        vis_data, _, vis_input_data = next(test_iterator)

                    vis_data = {k: fabric.to_device(v) for k, v in vis_data.items()}
                    vis_input_data = fabric.to_device(vis_input_data)

                    rot_transform_quats = vis_data["source_cv2wT_quat"][
                        :, : cfg.data.input_images
                    ]

                    if (
                        cfg.data.category == "hydrants"
                        or cfg.data.category == "teddybears"
                    ):
                        assert False
                        focals_pixels_pred = vis_data["focals_pixels"][
                            :, : cfg.data.input_images, ...
                        ]
                        input_images = torch.cat(
                            [
                                vis_data["gt_images"][:, : cfg.data.input_images, ...],
                                vis_data["origin_distances"][
                                    :, : cfg.data.input_images, ...
                                ],
                            ],
                            dim=2,
                        )
                    else:
                        focals_pixels_pred = None
                        input_images = adjust_channels(cfg, vis_input_data)

                    gaussian_splats_vis = gaussian_predictor(
                        input_images,
                        vis_data["view_to_world_transforms"][
                            :, : cfg.data.input_images, ...
                        ],
                        rot_transform_quats,
                        focals_pixels_pred,
                    )

                    test_loop = []
                    test_loop_gt = []
                    for r_idx in range(vis_data["gt_images"].shape[1]):
                        # We don't change the input or output of the network, just the rendering cameras
                        if "focals_pixels" in vis_data.keys():
                            focals_pixels_render = vis_data["focals_pixels"][0, r_idx]
                        else:
                            focals_pixels_render = None
                        test_image = render_predicted(
                            {
                                k: v[0].contiguous()
                                for k, v in gaussian_splats_vis.items()
                            },
                            vis_data["world_view_transforms"][0, r_idx],
                            vis_data["full_proj_transforms"][0, r_idx],
                            vis_data["camera_centers"][0, r_idx],
                            background,
                            cfg,
                            focals_pixels=focals_pixels_render,
                        )["render"]
                        test_loop_gt.append(
                            (
                                np.clip(
                                    vis_data["gt_images"][0, r_idx]
                                    .detach()
                                    .cpu()
                                    .numpy(),
                                    0,
                                    1,
                                )
                                * 255
                            ).astype(np.uint8)
                        )
                        test_loop.append(
                            (
                                np.clip(test_image.detach().cpu().numpy(), 0, 1) * 255
                            ).astype(np.uint8)
                        )

                    wandb.log(
                        {
                            "rot": wandb.Video(
                                np.asarray(test_loop), fps=20, format="mp4"
                            )
                        },
                        step=iteration,
                    )
                    wandb.log(
                        {
                            "rot_gt": wandb.Video(
                                np.asarray(test_loop_gt), fps=20, format="mp4"
                            )
                        },
                        step=iteration,
                    )

            fnames_to_save = []
            # Find out which models to save
            if (
                iteration + 1
            ) % cfg.logging.ckpt_iterations == 0 and fabric.is_global_zero:
                fnames_to_save.append("model_latest.pth")
            if (iteration + 1) % cfg.logging.val_log == 0 and fabric.is_global_zero:
                torch.cuda.empty_cache()
                print("\n[ITER {}] Validating".format(iteration + 1))
                if cfg.opt.ema.use:
                    scores = evaluate_dataset(
                        ema,
                        val_dataloader,
                        device=device,
                        model_cfg=cfg,
                    )
                else:
                    scores = evaluate_dataset(
                        gaussian_predictor,
                        val_dataloader,
                        device=device,
                        model_cfg=cfg,
                    )
                wandb.log(scores, step=iteration + 1)
                # save models - if the newest psnr is better than the best one,
                # overwrite best_model. Always overwrite the latest model.
                if scores["PSNR_novel"] > best_PSNR:
                    fnames_to_save.append("model_best.pth")
                    best_PSNR = scores["PSNR_novel"]
                    print(
                        "\n[ITER {}] Saving new best checkpoint PSNR:{:.2f}".format(
                            iteration + 1, best_PSNR
                        )
                    )
                torch.cuda.empty_cache()

            # ============ Model saving =================
            for fname_to_save in fnames_to_save:
                ckpt_save_dict = {
                    "iteration": iteration,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss.item(),
                    "best_PSNR": best_PSNR,
                }
                if cfg.opt.ema.use:
                    ckpt_save_dict["model_state_dict"] = ema.ema_model.state_dict()
                else:
                    ckpt_save_dict["model_state_dict"] = gaussian_predictor.state_dict()
                torch.save(ckpt_save_dict, os.path.join(vis_dir, fname_to_save))

            gaussian_predictor.train()

    wandb_run.finish()


if __name__ == "__main__":
    main()
