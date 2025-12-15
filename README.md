# medical sam2 - nyu deep learning project

hey so this is my project for nyu deep learning class. basically trying to adapt sam2 (segment anything model 2) for medical imaging stuff. works pretty well honestly, sometimes crashes but that's debugging for you.

## requirements (or whatever)

first things first, get your environment set up:

``conda env create -f environment.yml``

``conda activate medsam2``

then grab the sam2 checkpoint from the checkpoints folder:

``bash checkpoints/download_ckpts.sh``

**note:** tested this on ubuntu 22.04 with conda 23.7.4 and python 3.12.4. if you're on macos or windows you might run into some weird issues with cuda stuff, good luck with that. also make sure you have a gpu otherwise training will take forever.

## quick start examples

okay so here's how to actually use this thing. you'll need some data first obviously.

### 2d case - refuge optic cup segmentation

this one's pretty straightforward. fundus images, segment the optic cup. classic medical imaging task.

**step 1:** get the refuge dataset. you can download it manually from huggingface or just use wget like a normal person:

``wget https://huggingface.co/datasets/jiayuanz3/REFUGE/resolve/main/REFUGE.zip``

``unzip REFUGE.zip``

make sure to put it in a `data` folder (create it if it doesn't exist). the code expects things in specific places and will complain if stuff is missing. wait here you go you can use this direclty :





**step 2:** run training. here are commands for all model sizes:

**hiera-tiny (fastest, ~6gb vram):**
```bash
python train_2d.py -net sam2 -exp_name REFUGE_MedSAM2_Tiny -vis 1 -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt -sam_config sam2_hiera_t -image_size 1024 -out_size 1024 -b 4 -val_freq 1 -dataset REFUGE -data_path ./data/REFUGE
```

**hiera-small (default, ~8gb vram):**
```bash
python train_2d.py -net sam2 -exp_name REFUGE_MedSAM2_Small -vis 1 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -out_size 1024 -b 4 -val_freq 1 -dataset REFUGE -data_path ./data/REFUGE
```

**hiera-base-plus (~12gb vram):**
```bash
python train_2d.py -net sam2 -exp_name REFUGE_MedSAM2_BasePlus -vis 1 -sam_ckpt ./checkpoints/sam2_hiera_base_plus.pt -sam_config sam2_hiera_b+ -image_size 1024 -out_size 1024 -b 4 -val_freq 1 -dataset REFUGE -data_path ./data/REFUGE
```

**hiera-large (best accuracy, ~20gb vram):**
```bash
python train_2d.py -net sam2 -exp_name REFUGE_MedSAM2_Large -vis 1 -sam_ckpt ./checkpoints/sam2_hiera_large.pt -sam_config sam2_hiera_l -image_size 1024 -out_size 1024 -b 4 -val_freq 1 -dataset REFUGE -data_path ./data/REFUGE
```

the `-vis 1` flag will save some visualization images so you can actually see what's happening. useful for debugging when things go wrong (which they will).

### 3d case - abdominal organ segmentation

this one's more fun, 3d volumes and multiple organs. btcv dataset has like 13 different organs to segment.

**step 1:** download btcv dataset:

``wget https://huggingface.co/datasets/jiayuanz3/btcv/resolve/main/btcv.zip``

``unzip btcv.zip``

again, put it in the `data` folder. organization matters here.

```# Create data directory if it doesn't exist
mkdir -p data

# Download the dataset directly into the data folder
wget -P data https://huggingface.co/datasets/jiayuanz3/btcv/resolve/main/btcv.zip

# Unzip in the data folder and then remove the zip file
cd data
unzip btcv.zip
rm btcv.zip
cd ..
```

**step 2:** train on 3d data. here are commands for all model sizes:

**hiera-tiny (fastest, ~6gb vram):**
```bash
python train_3d.py -net sam2 -exp_name BTCV_MedSAM2_Tiny -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt -sam_config sam2_hiera_t -image_size 1024 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset btcv -data_path ./data/btcv
```

**hiera-small (default, ~8gb vram):**
```bash
python train_3d.py -net sam2 -exp_name BTCV_MedSAM2_Small -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset btcv -data_path ./data/btcv
```

**hiera-base-plus (~12gb vram):**
```bash
python train_3d.py -net sam2 -exp_name BTCV_MedSAM2_BasePlus -sam_ckpt ./checkpoints/sam2_hiera_base_plus.pt -sam_config sam2_hiera_b+ -image_size 1024 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset btcv -data_path ./data/btcv
```

**hiera-large (best accuracy, ~20gb vram):**
```bash
python train_3d.py -net sam2 -exp_name BTCV_MedSAM2_Large -sam_ckpt ./checkpoints/sam2_hiera_large.pt -sam_config sam2_hiera_l -image_size 1024 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset btcv -data_path ./data/btcv
```

the `-prompt bbox` means we're using bounding box prompts (you can also use points or masks). `-prompt_freq 2` means every 2 epochs we'll do some prompt-based validation. play around with these if you want.

**note:** if you have a powerful gpu (a100, h100, etc), you can increase batch size (`-b`) for larger models. try `-b 8` or `-b 16` for large model if you have the vram.

## inference (running predictions after training)

so you've trained a model and want to actually use it. here's how:

**for 3d volumes:**

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

the script will:
- load your trained checkpoint
- run predictions on test data
- save prediction masks as .npy files
- optionally save visualizations if you use `-save_vis`

outputs go to `inference_results/` by default. each sample gets its own folder with predictions for each frame.

**key args:**
- `-checkpoint`: path to your trained model (usually `best_dice_epoch.pth` or `latest_epoch.pth`)
- `-sam_config`: must match what you used for training (t/s/b+/l)
- `-data_path`: where your test data lives
- `-output_dir`: where to save results
- `-save_vis`: adds this flag to generate side-by-side comparison images

the checkpoint should be in your logs directory, something like `logs/BTCV_MedSAM2_2025_12_14_22_54_28/Model/best_dice_epoch.pth`. wandb will also log the best checkpoint path if you're using it.

## random notes and things that might help

- the codebase is kind of messy, sorry about that. it started clean but then experiments happened.
- if you're running out of gpu memory, try reducing batch size (`-b`) or image size. 1024x1024 is already pretty big.
- checkpoints get saved automatically, look in whatever directory you specify with `-exp_name`.
- there's also a colab notebook (`colab_medsam2.ipynb`) if you want to try things out without setting up the whole environment.
- the `func_2d` and `func_3d` folders have the actual training logic. `sam2_train` has the model architecture stuff.
- validation happens every `-val_freq` epochs. set it to 1 if you want to see results after every epoch (slower but more info).
- if something breaks, check the error message carefully. most issues are either missing data, wrong paths, or cuda/device mismatches.

## what this actually does

so sam2 is meta's segment anything model. we're fine-tuning it on medical images because medical images are weird and different from natural images. the model learns to segment organs, lesions, anatomical structures, etc. 

for 2d: works on single images like fundus photos, x-rays, that kind of thing.
for 3d: handles volumetric data like ct scans, mri volumes. processes them slice by slice but maintains some 3d context.

the training uses a combination of the original sam2 loss and some medical-specific losses. check the function files if you want the gory details.

## model variants

you can use different sam2 models depending on your needs and hardware. all checkpoints get downloaded with the download script.

available models:
- **hiera-tiny** (sam2_hiera_t): 38m params, ~6gb vram. fastest, good for quick experiments.
- **hiera-small** (sam2_hiera_s): 46m params, ~8gb vram. default choice, best speed/accuracy balance.
- **hiera-base-plus** (sam2_hiera_b+): 80m params, ~12gb vram. higher accuracy, still reasonably fast.
- **hiera-large** (sam2_hiera_l): 224m params, ~20gb vram. best accuracy, slower training.

complete training commands for all models are shown above in the quick start examples. main differences: larger models have more transformer blocks and wider embeddings. tiny has 12 blocks, small has 16, base+ has 24, large has 48. more blocks = better accuracy but slower training and more memory. if you have a beefy gpu (a100, h100, etc), go for large. if memory is tight, stick with small or tiny.

see `MODEL_COMPARISON.md` for detailed specs and benchmarks.

**quick way to train both tiny and small together:**

if you want to train both tiny and small models sequentially (useful for comparing results), use the shell script:

```bash
bash train_both_models.sh
```

or with custom options:

```bash
bash train_both_models.sh --dataset btcv --data-path ./data/btcv --batch-size 2 --image-size 1024
```

the script runs tiny first, then small. checkpoints get saved separately in their respective log directories.


## troubleshooting

- "cuda out of memory": reduce batch size or image size, or use a smaller model variant
- "file not found": check your data paths, make sure folders exist
- "module not found": did you activate the conda environment?
- weird segmentation results: check your data preprocessing, maybe the normalization is off
- training loss not decreasing: try different learning rates, check if your data is actually loading correctly

