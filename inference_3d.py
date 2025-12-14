#!/usr/bin/env python3
"""
inference script for 3d medical image segmentation
load trained checkpoint and run predictions on new data
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sam2_train.build_sam import build_sam2_video_predictor
from func_3d.dataset import get_dataloader
import cfg


def parse_inference_args():
    parser = argparse.ArgumentParser(description='inference for medical sam2 3d')
    
    # model args
    parser.add_argument('-checkpoint', type=str, required=True,
                        help='path to trained checkpoint (e.g., logs/BTCV_MedSAM2_*/Model/best_dice_epoch.pth)')
    parser.add_argument('-sam_config', type=str, default='sam2_hiera_s',
                        choices=['sam2_hiera_t', 'sam2_hiera_s', 'sam2_hiera_b+', 'sam2_hiera_l'],
                        help='sam2 config to use')
    
    # data args
    parser.add_argument('-data_path', type=str, required=True,
                        help='path to test data (e.g., ./data/btcv)')
    parser.add_argument('-dataset', type=str, default='btcv',
                        help='dataset name')
    parser.add_argument('-prompt', type=str, default='bbox',
                        choices=['click', 'bbox'],
                        help='prompt type')
    parser.add_argument('-prompt_freq', type=int, default=2,
                        help='prompt frequency')
    
    # output args
    parser.add_argument('-output_dir', type=str, default='./inference_results',
                        help='where to save inference results')
    parser.add_argument('-save_vis', action='store_true',
                        help='save visualization images')
    parser.add_argument('-gpu_device', type=int, default=0,
                        help='gpu device id')
    
    # other args that get_dataloader expects
    parser.add_argument('-image_size', type=int, default=1024)
    parser.add_argument('-video_length', type=int, default=2)
    parser.add_argument('-b', type=int, default=1)
    parser.add_argument('-net', type=str, default='sam2')
    
    return parser.parse_args()


def load_trained_model(checkpoint_path, sam_config, device):
    """load model with trained weights"""
    print(f"loading model from {checkpoint_path}")
    
    # build base model
    model = build_sam2_video_predictor(
        config_file=sam_config,
        ckpt_path=None,  # don't load pretrained weights yet
        device=device,
        mode='eval'
    )
    
    # load trained checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("checkpoint loaded successfully")
    else:
        raise FileNotFoundError(f"checkpoint not found at {checkpoint_path}")
    
    model.to(dtype=torch.bfloat16)
    model.eval()
    return model


def run_inference(model, test_loader, args, device):
    """run inference on test data"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print(f"running inference on {len(test_loader)} samples...")
    
    with torch.no_grad():
        for idx, pack in enumerate(test_loader):
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            
            if args.prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif args.prompt == 'bbox':
                bbox_dict = pack['bbox']
            
            if len(imgs_tensor.size()) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)
            
            imgs_tensor = imgs_tensor.to(dtype=torch.float32, device=device)
            frame_id = list(range(imgs_tensor.size(0)))
            
            # init inference state
            inference_state = model.val_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, len(frame_id), args.prompt_freq))
            
            # get object list
            obj_list = []
            for id in frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            
            if len(obj_list) == 0:
                print(f"skipping sample {idx}: no objects")
                continue
            
            name = pack['image_meta_dict']['filename_or_obj'][0]
            print(f"processing {name}...")
            
            # add prompts
            for id in prompt_frame_id:
                for ann_obj_id in obj_list:
                    try:
                        if args.prompt == 'click':
                            points = pt_dict[id][ann_obj_id].to(device=device)
                            labels = point_labels_dict[id][ann_obj_id].to(device=device)
                            _, _, _ = model.train_add_new_points(
                                inference_state=inference_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                points=points,
                                labels=labels,
                                clear_old_points=False,
                            )
                        elif args.prompt == 'bbox':
                            bbox = bbox_dict[id][ann_obj_id]
                            _, _, _ = model.train_add_new_bbox(
                                inference_state=inference_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                bbox=bbox.to(device=device),
                                clear_old_points=False,
                            )
                    except KeyError:
                        # no prompt for this frame/object, use empty mask
                        _, _, _ = model.train_add_new_mask(
                            inference_state=inference_state,
                            frame_idx=id,
                            obj_id=ann_obj_id,
                            mask=torch.zeros(imgs_tensor.shape[2:]).to(device=device),
                        )
            
            # propagate and get predictions
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(inference_state, start_frame_idx=0):
                video_segments[out_frame_idx] = {
                    out_obj_id: out_mask_logits[i]
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            # save predictions
            sample_result = {
                'name': name,
                'predictions': {},
                'ground_truth': {}
            }
            
            for id in frame_id:
                for ann_obj_id in obj_list:
                    pred = video_segments[id][ann_obj_id]
                    pred_binary = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                    
                    # save prediction
                    pred_dir = output_dir / name / f"frame_{id:03d}"
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    np.save(pred_dir / f"pred_obj_{ann_obj_id}.npy", pred_binary)
                    
                    # optionally save visualization
                    if args.save_vis:
                        try:
                            gt_mask = mask_dict[id][ann_obj_id].cpu().numpy()
                        except KeyError:
                            gt_mask = np.zeros_like(pred_binary)
                        
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        
                        # image
                        img_slice = imgs_tensor[id].cpu().permute(1, 2, 0).numpy()
                        if img_slice.shape[2] == 1:
                            axes[0].imshow(img_slice[:, :, 0], cmap='gray')
                        else:
                            axes[0].imshow(img_slice.astype(int))
                        axes[0].set_title('input image')
                        axes[0].axis('off')
                        
                        # prediction
                        axes[1].imshow(pred_binary, cmap='gray')
                        axes[1].set_title('prediction')
                        axes[1].axis('off')
                        
                        # ground truth
                        axes[2].imshow(gt_mask[0, 0] if len(gt_mask.shape) > 2 else gt_mask, cmap='gray')
                        axes[2].set_title('ground truth')
                        axes[2].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(pred_dir / f"vis_obj_{ann_obj_id}.png", dpi=150, bbox_inches='tight')
                        plt.close()
                    
                    sample_result['predictions'][(id, ann_obj_id)] = pred_binary
                    try:
                        sample_result['ground_truth'][(id, ann_obj_id)] = mask_dict[id][ann_obj_id].cpu().numpy()
                    except KeyError:
                        pass
            
            results.append(sample_result)
            model.reset_state(inference_state)
            
            print(f"completed {idx + 1}/{len(test_loader)}")
    
    print(f"\ninference complete! results saved to {output_dir}")
    return results


def main():
    args = parse_inference_args()
    
    # setup device
    device = torch.device(f'cuda:{args.gpu_device}' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    
    # load model
    model = load_trained_model(args.checkpoint, args.sam_config, device)
    
    # get test dataloader
    # hacky way to get dataloader - reuse training code
    args.gpu = True
    _, test_loader = get_dataloader(args)
    
    # run inference
    results = run_inference(model, test_loader, args, device)
    
    print(f"\nprocessed {len(results)} samples")
    print(f"check results in: {args.output_dir}")


if __name__ == '__main__':
    main()

