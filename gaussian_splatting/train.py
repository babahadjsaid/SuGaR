#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import gc, torch
import os
import numpy  # Added to avoid issue
import torch
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.camera_utils import loadCam
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# Import depth-normal utilities
from depth_normal_utils import (
    depth_normal_consistency_loss,
    render_depth_and_normal
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    train_dataloader = scene.getTrainCameras()  # Get DataLoader
    train_iter = iter(train_dataloader)  # Iterator for the DataLoader
    
    # Hardcoded intervals for depth-normal consistency and entropy regularization
    dn_consistency_intervals = [(6000, 6999), (29000, 29999)]
    entropy_reg_intervals = [(6900, 6999), (29900, 29999)]
    dn_consistency_factor = 0.1  # Weight for DN consistency loss
    entropy_reg_factor = 0.1  # Weight for entropy regularization
    
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        try:
            viewpoint_cam = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            viewpoint_cam = next(train_iter)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # Check if current iteration is in any DN consistency interval
        apply_dn_consistency = any(start <= iteration <= end for start, end in dn_consistency_intervals)
        
        # Add Depth-Normal consistency loss during specified intervals
        if apply_dn_consistency:
            try:
                # Render depth and normals using Gaussian rasterizer (proper blending approach)
                depth_img, normal_img = render_depth_and_normal(viewpoint_cam, gaussians, pipe, bg)
                
                # Only compute loss if we have valid depth and normals
                if depth_img.max() > 0 and normal_img.norm(dim=-1).max() > 0:
                    # Reshape for consistency loss function
                    depth_tensor = depth_img[None]  # Shape: (1, height, width)
                    normal_tensor = normal_img.permute(2, 0, 1)  # Shape: (3, height, width)
                    
                    # Compute depth-normal consistency loss
                    normal_error = depth_normal_consistency_loss(
                        depth=depth_tensor,
                        normal=normal_tensor,
                        camera=viewpoint_cam,
                        scale_rendered_normals=False,
                        return_normal_maps=False,
                    )
                    
                    # Add to total loss
                    loss = loss + dn_consistency_factor * normal_error
                    
                    # Log the DN consistency loss
                    if tb_writer and iteration % 100 == 0:
                        tb_writer.add_scalar('train_loss_patches/dn_consistency_loss', normal_error.item(), iteration)
                        
            except Exception as e:
                # If DN consistency fails, continue without it
                if iteration % 1000 == 0:  # Only print warning occasionally
                    print(f"Warning: DN consistency loss failed at iteration {iteration}: {e}")
                pass
        
        # Check if current iteration is in any entropy regularization interval
        apply_entropy_reg = any(start <= iteration <= end for start, end in entropy_reg_intervals)
        
        # Add entropy regularization for opacities during specified intervals
        if apply_entropy_reg:
            try:
                # Get opacities from gaussians
                opacities = gaussians.get_opacity()
                
                # Apply visibility filter if available
                if 'visibility_filter' in locals() and visibility_filter is not None:
                    vis_opacities = opacities[visibility_filter]
                else:
                    vis_opacities = opacities
                
                # Compute entropy regularization loss
                # L_entropy = -p*log(p) - (1-p)*log(1-p)
                entropy_loss = (
                    - vis_opacities * torch.log(vis_opacities + 1e-10)
                    - (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)
                ).mean()
                
                # Add to total loss
                loss = loss + entropy_reg_factor * entropy_loss
                
                # Log the entropy regularization loss
                if tb_writer and iteration % 100 == 0:
                    tb_writer.add_scalar('train_loss_patches/entropy_reg_loss', entropy_loss.item(), iteration)
                    
            except Exception as e:
                # If entropy regularization fails, continue without it
                if iteration % 1000 == 0:  # Only print warning occasionally
                    print(f"Warning: Entropy regularization failed at iteration {iteration}: {e}")
                pass
        
        loss.backward()

        iter_end.record()
        del gt_image, viewpoint_cam, image, render_pkg
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, dataset):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : list(scene.getTestCameras())}, 
                              {'name': 'train', 'cameras' : [loadCam(dataset, idx, scene.getTrainCamInfos()[idx % len(scene.getTrainCamInfos())], 1.0) for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        # remove the loaded image 
        del viewpoint, image, gt_image
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    torch.multiprocessing.set_sharing_strategy('file_system')  # Use files instead of SHM for data sharing
    
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
