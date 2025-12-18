# Medical SAM2-N: Adapting Segment Anything Model 2 for Medical Imaging

## Acknowledgements

**Note on Implementation:**
This project utilizes a modified version of the [Medical-SAM2 repository](https://github.com/ImprintLab/Medical-SAM2) developed by ImprintLaq as well as base notebooks created by Meta in their [SAM2 repository](https://github.com/facebookresearch/sam2?tab=readme-ov-file). We leveraged their codebase to facilitate our training runs; given the significant architectural complexity of SAM2, utilizing this optimized framework was necessary to ensure computational efficiency and project feasibility within our time and resource constraints.

## Overview

This repository implements an adaptation of the Segment Anything Model 2 (SAM2) specifically optimized for medical imaging tasks. The project demonstrates the fine-tuning of SAM2 for:

* **2D Segmentation:** Optical cup segmentation (fundus images).
* **3D Segmentation:** Multi-organ segmentation (CT/MRI volumetric data).

The architecture incorporates medical-specific loss functions and data pipelines to handle the unique characteristics of biomedical data compared to natural images.

## Installation & Requirements

### 1. Environment Setup

To establish the necessary dependencies, create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate medsam2

```

### 2. Model Checkpoints

Download the pre-trained SAM2 checkpoints required for initialization:

```bash
bash checkpoints/download_ckpts.sh

```

**System Compatibility:**

* **OS:** Validated on Ubuntu 22.04.
* **Python:** 3.12.4
* **Conda:** 23.7.4
* **Hardware:** A generic GPU is required for training. Users on macOS or Windows may encounter CUDA-related compatibility issues and may need to adjust dependencies accordingly.

---

## Dataset Preparation

Data must be organized within a root `data` directory.

### 2D Task: REFUGE (Optic Cup Segmentation)

Download the REFUGE dataset and extract it into the `data` directory:

```bash
# Create data directory if it doesn't exist
mkdir -p data

# Download and unzip
wget -P data https://huggingface.co/datasets/jiayuanz3/REFUGE/resolve/main/REFUGE.zip
cd data
unzip REFUGE.zip
rm REFUGE.zip
cd ..

```

### 3D Task: BTCV (Abdominal Organ Segmentation)

Download the BTCV dataset (containing ~13 organs for segmentation) and extract it:

```bash
# Download and unzip
wget -P data https://huggingface.co/datasets/jiayuanz3/btcv/resolve/main/btcv.zip
cd data
unzip btcv.zip
rm btcv.zip
cd ..

```

---

## Training

### 2D Training Instructions

The following commands fine-tune the model on the REFUGE dataset. The `-vis 1` flag enables the saving of visualization images for debugging purposes.

**Hiera-Tiny (Fastest, ~6GB VRAM):**

```bash
python train_2d.py -net sam2 -exp_name REFUGE_MedSAM2_Tiny -vis 1 -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt -sam_config sam2_hiera_t -image_size 1024 -out_size 1024 -b 4 -val_freq 1 -dataset REFUGE -data_path ./data/REFUGE

```

**Hiera-Small (Default, ~8GB VRAM):**

```bash
python train_2d.py -net sam2 -exp_name REFUGE_MedSAM2_Small -vis 1 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -out_size 1024 -b 4 -val_freq 1 -dataset REFUGE -data_path ./data/REFUGE

```

**Hiera-Base-Plus (~12GB VRAM):**

```bash
python train_2d.py -net sam2 -exp_name REFUGE_MedSAM2_BasePlus -vis 1 -sam_ckpt ./checkpoints/sam2_hiera_base_plus.pt -sam_config sam2_hiera_b+ -image_size 1024 -out_size 1024 -b 4 -val_freq 1 -dataset REFUGE -data_path ./data/REFUGE

```

**Hiera-Large (Best Accuracy, ~20GB VRAM):**

```bash
python train_2d.py -net sam2 -exp_name REFUGE_MedSAM2_Large -vis 1 -sam_ckpt ./checkpoints/sam2_hiera_large.pt -sam_config sam2_hiera_l -image_size 1024 -out_size 1024 -b 4 -val_freq 1 -dataset REFUGE -data_path ./data/REFUGE

```

### 3D Training Instructions

The following commands fine-tune the model on the BTCV volumetric dataset.

**Key Arguments:**

* `-prompt bbox`: Utilizes bounding box prompts (points and masks are also supported).
* `-prompt_freq 2`: performs prompt-based validation every 2 epochs.
* **Batch Size (`-b`):** If utilizing high-end GPUs (e.g., A100, H100), consider increasing the batch size (`-b 8` or `-b 16`) for improved throughput.

**Hiera-Tiny (Fastest, ~6GB VRAM):**

```bash
python train_3d.py -net sam2 -exp_name BTCV_MedSAM2_Tiny -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt -sam_config sam2_hiera_t -image_size 1024 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset btcv -data_path ./data/btcv

```

**Hiera-Small (Default, ~8GB VRAM):**

```bash
python train_3d.py -net sam2 -exp_name BTCV_MedSAM2_Small -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset btcv -data_path ./data/btcv

```

**Hiera-Base-Plus (~12GB VRAM):**

```bash
python train_3d.py -net sam2 -exp_name BTCV_MedSAM2_BasePlus -sam_ckpt ./checkpoints/sam2_hiera_base_plus.pt -sam_config sam2_hiera_b+ -image_size 1024 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset btcv -data_path ./data/btcv

```

**Hiera-Large (Best Accuracy, ~20GB VRAM):**

```bash
python train_3d.py -net sam2 -exp_name BTCV_MedSAM2_Large -sam_ckpt ./checkpoints/sam2_hiera_large.pt -sam_config sam2_hiera_l -image_size 1024 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset btcv -data_path ./data/btcv

```

### Batch Training Script

To train both Tiny and Small models sequentially (useful for comparative analysis), utilize the provided shell script:

```bash
bash train_both_models.sh --dataset btcv --data-path ./data/btcv --batch-size 2 --image-size 1024

```

**High-Memory Environment (A100/H100/H200):**
Use the `--high-memory` flag to maximize utilization (sets batch size to 16 for Tiny and 12 for Small):

```bash
bash train_both_models.sh --high-memory

```

---

## Inference

To run predictions on a test set using a trained checkpoint:

```bash
python inference_3d.py \
  -checkpoint ./logs/BTCV_MedSAM2_*/Model/best_dice_epoch.pth \
  -sam_config sam2_hiera_s \
  -data_path ./data/btcv \
  -dataset btcv \
  -prompt bbox \
  -output_dir ./inference_results \
  -save_vis

```

**Workflow:**

1. Loads the trained checkpoint specified by `-checkpoint`.
2. Runs predictions on the test split.
3. Saves prediction masks as `.npy` files in the `output_dir`.
4. (Optional) Generates side-by-side visualizations if `-save_vis` is enabled.

**Note:** Ensure `-sam_config` matches the architecture used during training.

---

## Model Variants & Performance

We support four variants of the Hiera backbone, allowing users to balance computational resources against segmentation accuracy. See `MODEL_COMPARISON.md` for detailed benchmarks.

| Model Variant | Config Name | Parameters | Est. VRAM | Use Case |
| --- | --- | --- | --- | --- |
| **Hiera-Tiny** | `sam2_hiera_t` | 38M | ~6GB | Rapid prototyping, low resource environments. |
| **Hiera-Small** | `sam2_hiera_s` | 46M | ~8GB | Default choice; balanced speed/accuracy. |
| **Hiera-Base+** | `sam2_hiera_b+` | 80M | ~12GB | Higher accuracy, moderate resource usage. |
| **Hiera-Large** | `sam2_hiera_l` | 224M | ~20GB | Maximum accuracy; requires high-end GPUs. |

*Larger models contain more transformer blocks (Tiny: 12, Small: 16, Base+: 24, Large: 48) and wider embeddings.*

---

## Implementation Details & Troubleshooting

### Project Structure

* **`func_2d/` & `func_3d/**`: Contains the core training loops and validation logic.
* **`sam2_train/`**: Contains the model architecture definitions.
* **`colab_medsam2.ipynb`**: A Google Colab notebook for testing the pipeline without local environment setup.

### Troubleshooting Guide

* **CUDA Out of Memory:** Reduce the batch size (`-b`), reduce image size (default is 1024), or switch to a smaller model variant (e.g., Tiny).
* **File Not Found Errors:** Verify that the `data` directory structure matches the hierarchy expected by the loader.
* **Poor Segmentation Results:** Investigate data preprocessing steps, particularly normalization. Ensure the loss function is converging.
* **Training Stalls:** Check if the learning rate is appropriate for the chosen batch size.
* **Validation:** Validation runs every `-val_freq` epochs. For granular debugging, set `-val_freq 1`.


---

# Traditional Baselines

To ensure rigorous performance evaluation, we provide implementations of standard computer vision architectures to serve as benchmarks against the SAM2 adaptation. These models are located in the `traditional_models` directory and include:

* **ResNet:** Standard residual network implementations for comparative segmentation baselines.
* **YOLO:** You Only Look Once architectures adapted for specific localization tasks.

These baselines allow for direct A/B testing of the SAM2-based adaptation against established industry standards.

## HPC Usage (Slurm & Singularity)

For users running training jobs on High Performance Computing (HPC) clusters (specifically configured for Slurm workloads with Singularity support), we provide a submission script to streamline the process.

### Running with Sbatch

The provided Slurm script accepts the model type as a command-line argument to dynamically configure the training job.

**Supported Model Arguments:**

* `resnet`
* `yolo`

**Submission Command:**
To submit a job, use `sbatch` followed by your script name and the desired model variant:

```bash
sbatch train_slurm.sh <model_name>

```

**Example:**

```bash
# Train the YoloV11L model
sbatch train_slurm.sh yolo

# Train the ResNet baseline
sbatch train_slurm.sh resnet

```

### Running Locally (Python Direct)

If you are running the training script directly in a local environment (bypassing the Slurm/Singularity wrapper), use the standard Python command:

```bash
python train.py --model <selected_model>

```

**Example:**

```bash
python train.py --model yolo

```

