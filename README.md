# medical sam2 - nyu deep learning project

hey so this is OUR  project for nyu deep learning class. basically trying to adapt sam2 (segment anything model 2) for medical imaging stuff. works pretty well honestly, sometimes crashes but that's debugging for you.

## requirements 

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

make sure to put it in a `data` folder (create it if it doesn't exist). the code expects things in specific places and will complain if stuff is missing.

**step 2:** run training:

``python train_2d.py -net sam2 -exp_name REFUGE_MedSAM2 -vis 1 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -out_size 1024 -b 4 -val_freq 1 -dataset REFUGE -data_path ./data/REFUGE``

the `-vis 1` flag will save some visualization images so you can actually see what's happening. useful for debugging when things go wrong (which they will).

### 3d case - abdominal organ segmentation

this one's more fun, 3d volumes and multiple organs. btcv dataset has like 13 different organs to segment.

**step 1:** download btcv dataset:

``wget https://huggingface.co/datasets/jiayuanz3/btcv/resolve/main/btcv.zip``

``unzip btcv.zip``

again, put it in the `data` folder. organization matters here. actually its part of the data dir itself, make sure when you run the train3d run command you map the check perfeclty, you can use ls -la to check all. /workspace/sam3-dl/Medical-SAM2/data/btc, its inside btc for the training script to work make sure of that

**step 2:** train on 3d data:

``python train_3d.py -net sam2 -exp_name BTCV_MedSAM2 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset btcv -data_path ./data/btc``

the `-prompt bbox` means we're using bounding box prompts (you can also use points or masks). `-prompt_freq 2` means every 2 epochs we'll do some prompt-based validation. play around with these if you want.

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

## troubleshooting

- "cuda out of memory": reduce batch size or image size, i alos used an h200 sxm so training was much faster
- "file not found": check your data paths, make sure folders exist
- "module not found": did you activate the conda environment?
- weird segmentation results: check your data preprocessing, maybe the normalization is off
- training loss not decreasing: try different learning rates, check if your data is actually loading correctly

