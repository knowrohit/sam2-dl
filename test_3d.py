#!/usr/bin/env python3
"""
Test script for Medical SAM 2 on 3D medical images (BTCV dataset)
Tests trained model on test set and saves predictions + metrics

Usage:
    python test_3d.py -weights ./logs/btcv_MedSAM2_Small_2025_12_15_21_15_05/Model/best_dice_epoch.pth \
                      -sam_ckpt ./checkpoints/sam2_hiera_small.pt \
                      -sam_config sam2_hiera_s \
                      -data_path ./data/btcv
"""

import os
import argparse
import time
import json
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import cfg
from func_3d import function
from conf import settings
from func_3d.utils import get_network, eval_seg
from func_3d.dataset import get_dataloader


def parse_test_args():
    parser = argparse.ArgumentParser(description='Test Medical SAM 2 on 3D medical images')
    
    # Model arguments
    parser.add_argument('-weights', type=str, required=True, help='Path to trained model weights (.pth file)')
    parser.add_argument('-sam_ckpt', type=str, required=True, help='Path to SAM2 checkpoint')
    parser.add_argument('-sam_config', type=str, required=True, help='SAM2 config (sam2_hiera_t, sam2_hiera_s, etc.)')
    
    # Data arguments
    parser.add_argument('-data_path', type=str, default='./data/btcv', help='Path to test data')
    parser.add_argument('-dataset', type=str, default='btcv', help='Dataset name (btcv or amos)')
    
    # Test configuration
    parser.add_argument('-prompt', type=str, default='bbox', help='Prompt type: bbox or click')
    parser.add_argument('-prompt_freq', type=int, default=2, help='Frequency of prompts in test (every N slices)')
    parser.add_argument('-image_size', type=int, default=1024, help='Input image size')
    parser.add_argument('-gpu_device', type=int, default=0, help='GPU device ID')
    parser.add_argument('-vis', type=int, default=10, help='Visualize every N samples (0 to disable)')
    parser.add_argument('-save_predictions', type=bool, default=True, help='Save prediction masks')
    
    # Output
    parser.add_argument('-output_dir', type=str, default='./test_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set additional required args for compatibility
    args.net = 'sam2'
    args.gpu = True
    args.distributed = 'none'
    args.video_length = None  # Will use full length for testing
    
    return args


def save_visualization(img_tensor, pred_masks, gt_masks, obj_ids, save_path, frame_ids=None):
    """Save visualization of predictions vs ground truth for a 3D volume"""
    
    if frame_ids is None:
        # Select 6 evenly spaced frames
        num_frames = img_tensor.shape[0]
        frame_ids = np.linspace(0, num_frames-1, min(6, num_frames), dtype=int)
    
    num_frames_to_show = len(frame_ids)
    fig, axes = plt.subplots(num_frames_to_show, 3, figsize=(12, 4*num_frames_to_show))
    
    if num_frames_to_show == 1:
        axes = axes.reshape(1, -1)
    
    for idx, frame_id in enumerate(frame_ids):
        # Image
        img = img_tensor[frame_id].cpu().permute(1, 2, 0).numpy().astype(int)
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'Frame {frame_id} - Input')
        axes[idx, 0].axis('off')
        
        # Prediction - combine all objects
        pred_combined = np.zeros(img_tensor.shape[2:])
        for obj_id in obj_ids:
            if frame_id in pred_masks and obj_id in pred_masks[frame_id]:
                pred_mask = (pred_masks[frame_id][obj_id].cpu().numpy() > 0.5).astype(int)
                pred_combined[pred_mask[0, 0] > 0] = obj_id
        
        axes[idx, 1].imshow(pred_combined, cmap='tab20')
        axes[idx, 1].set_title(f'Frame {frame_id} - Prediction')
        axes[idx, 1].axis('off')
        
        # Ground truth - combine all objects
        gt_combined = np.zeros(img_tensor.shape[2:])
        for obj_id in obj_ids:
            if frame_id in gt_masks and obj_id in gt_masks[frame_id]:
                gt_mask = gt_masks[frame_id][obj_id].cpu().numpy().astype(int)
                gt_combined[gt_mask[0, 0] > 0] = obj_id
        
        axes[idx, 2].imshow(gt_combined, cmap='tab20')
        axes[idx, 2].set_title(f'Frame {frame_id} - Ground Truth')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def test_sam(args, test_loader, net, output_dir):
    """Test the model on the test set"""
    
    # Set up
    net.eval()
    GPUdevice = torch.device('cuda', args.gpu_device)
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    prompt = args.prompt
    prompt_freq = args.prompt_freq
    
    # Metrics storage
    all_results = []
    total_iou = 0
    total_dice = 0
    total_samples = 0
    
    # Loss function
    criterion_G = torch.nn.BCEWithLogitsLoss()
    total_loss = 0
    
    # Output directories
    vis_dir = os.path.join(output_dir, 'visualizations')
    pred_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(vis_dir, exist_ok=True)
    if args.save_predictions:
        os.makedirs(pred_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Testing Medical SAM 2 - {args.sam_config}")
    print(f"Weights: {args.weights}")
    print(f"Dataset: {args.dataset}")
    print(f"Prompt: {args.prompt} (every {args.prompt_freq} slices)")
    print(f"{'='*70}\n")
    
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Testing', unit='volume') as pbar:
            for idx, pack in enumerate(test_loader):
                imgs_tensor = pack['image']
                mask_dict = pack['label']
                
                if prompt == 'click':
                    pt_dict = pack['pt']
                    point_labels_dict = pack['p_label']
                elif prompt == 'bbox':
                    bbox_dict = pack['bbox']
                
                if len(imgs_tensor.size()) == 5:
                    imgs_tensor = imgs_tensor.squeeze(0)
                
                num_frames = imgs_tensor.size(0)
                frame_ids = list(range(num_frames))
                
                # Initialize inference state
                inference_state = net.val_init_state(imgs_tensor=imgs_tensor)
                
                # Get prompt frames
                prompt_frame_ids = list(range(0, len(frame_ids), prompt_freq))
                
                # Get all objects in this volume
                obj_list = []
                for frame_id in frame_ids:
                    obj_list += list(mask_dict[frame_id].keys())
                obj_list = list(set(obj_list))
                
                if len(obj_list) == 0:
                    pbar.update()
                    continue
                
                name = pack['image_meta_dict']['filename_or_obj'][0]
                
                # Add prompts to specified frames
                for frame_id in prompt_frame_ids:
                    for obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[frame_id][obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[frame_id][obj_id].to(device=GPUdevice)
                                _, _, _ = net.train_add_new_points(
                                    inference_state=inference_state,
                                    frame_idx=frame_id,
                                    obj_id=obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[frame_id][obj_id]
                                _, _, _ = net.train_add_new_bbox(
                                    inference_state=inference_state,
                                    frame_idx=frame_id,
                                    obj_id=obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            # Object not present in this frame
                            _, _, _ = net.train_add_new_mask(
                                inference_state=inference_state,
                                frame_idx=frame_id,
                                obj_id=obj_id,
                                mask=torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                
                # Propagate through all frames
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(
                    inference_state, start_frame_idx=0
                ):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                
                # Calculate metrics for this volume
                volume_loss = 0
                volume_iou = 0
                volume_dice = 0
                volume_count = 0
                
                # Store predictions for this volume
                volume_predictions = {}
                
                for frame_id in frame_ids:
                    for obj_id in obj_list:
                        pred = video_segments[frame_id][obj_id]
                        pred = pred.unsqueeze(0)
                        
                        try:
                            mask = mask_dict[frame_id][obj_id].to(dtype=torch.float32, device=GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        
                        # Calculate loss
                        volume_loss += criterion_G(pred, mask).item()
                        
                        # Calculate metrics
                        pred_binary = (pred > 0.5).float()
                        temp = eval_seg(pred_binary, mask, threshold)
                        volume_iou += temp[0]
                        volume_dice += temp[1]
                        volume_count += 1
                        
                        # Store prediction
                        if frame_id not in volume_predictions:
                            volume_predictions[frame_id] = {}
                        volume_predictions[frame_id][obj_id] = pred.cpu()
                
                # Average metrics for this volume
                volume_loss /= volume_count
                volume_iou /= volume_count
                volume_dice /= volume_count
                
                total_loss += volume_loss
                total_iou += volume_iou
                total_dice += volume_dice
                total_samples += 1
                
                # Store results
                result = {
                    'name': name,
                    'num_frames': num_frames,
                    'num_objects': len(obj_list),
                    'loss': volume_loss,
                    'iou': volume_iou,
                    'dice': volume_dice,
                }
                all_results.append(result)
                
                # Save predictions
                if args.save_predictions:
                    pred_save_path = os.path.join(pred_dir, name)
                    os.makedirs(pred_save_path, exist_ok=True)
                    for frame_id in volume_predictions:
                        for obj_id in volume_predictions[frame_id]:
                            pred_mask = (volume_predictions[frame_id][obj_id].numpy() > 0.5).astype(np.uint8)
                            np.save(
                                os.path.join(pred_save_path, f'frame_{frame_id}_obj_{obj_id}.npy'),
                                pred_mask
                            )
                
                # Visualize
                if args.vis > 0 and idx % args.vis == 0:
                    vis_path = os.path.join(vis_dir, f'{name}_visualization.png')
                    save_visualization(
                        imgs_tensor, 
                        volume_predictions, 
                        mask_dict, 
                        obj_list, 
                        vis_path
                    )
                
                # Reset state
                net.reset_state(inference_state)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{volume_loss:.4f}',
                    'IoU': f'{volume_iou:.4f}',
                    'Dice': f'{volume_dice:.4f}'
                })
                pbar.update()
    
    # Calculate final metrics
    avg_loss = total_loss / total_samples
    avg_iou = total_iou / total_samples
    avg_dice = total_dice / total_samples
    
    return avg_loss, avg_iou, avg_dice, all_results


def main():
    # Parse arguments
    args = parse_test_args()
    
    # Setup device
    GPUdevice = torch.device('cuda', args.gpu_device)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = os.path.basename(os.path.dirname(args.weights))
    output_dir = os.path.join(args.output_dir, f'{model_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}\n")
    
    # Load model
    print("Loading model...")
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    net.to(dtype=torch.bfloat16)
    
    # Load trained weights
    if os.path.exists(args.weights):
        print(f"Loading weights from: {args.weights}")
        checkpoint = torch.load(args.weights, map_location=GPUdevice)
        if 'model' in checkpoint:
            net.load_state_dict(checkpoint['model'], strict=False)
        else:
            net.load_state_dict(checkpoint, strict=False)
        print("✓ Weights loaded successfully\n")
    else:
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    
    # Load test data
    print("Loading test data...")
    _, test_loader = get_dataloader(args)
    print(f"✓ Test set loaded: {len(test_loader)} volumes\n")
    
    # Run testing
    start_time = time.time()
    avg_loss, avg_iou, avg_dice, all_results = test_sam(args, test_loader, net, output_dir)
    test_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*70}")
    print(f"TEST RESULTS")
    print(f"{'='*70}")
    print(f"Average Loss:  {avg_loss:.4f}")
    print(f"Average IoU:   {avg_iou:.4f}")
    print(f"Average Dice:  {avg_dice:.4f}")
    print(f"Test Time:     {test_time:.2f} seconds")
    print(f"{'='*70}\n")
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'test_results.json')
    results_summary = {
        'model': {
            'weights': args.weights,
            'sam_config': args.sam_config,
            'sam_checkpoint': args.sam_ckpt,
        },
        'test_config': {
            'dataset': args.dataset,
            'data_path': args.data_path,
            'prompt': args.prompt,
            'prompt_freq': args.prompt_freq,
            'image_size': args.image_size,
        },
        'metrics': {
            'average_loss': float(avg_loss),
            'average_iou': float(avg_iou),
            'average_dice': float(avg_dice),
            'num_test_samples': len(all_results),
            'test_time_seconds': float(test_time),
        },
        'per_volume_results': all_results,
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"✓ Detailed results saved to: {results_file}")
    print(f"✓ Visualizations saved to: {os.path.join(output_dir, 'visualizations')}")
    if args.save_predictions:
        print(f"✓ Predictions saved to: {os.path.join(output_dir, 'predictions')}")
    
    # Save per-volume results as CSV
    import pandas as pd
    df = pd.DataFrame(all_results)
    csv_file = os.path.join(output_dir, 'test_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"✓ CSV results saved to: {csv_file}\n")


if __name__ == '__main__':
    main()

