from extract import extract_dataset_archive
from split import convert_train_set_to_images
import os

def prepare_dataset():
    extract_dataset_archive()
    convert_train_set_to_images()


prepare_dataset()
os.chdir("AlphaPose")
os.system('python scripts/demo_inference.py --cfg pretrained_models/256x192_res152_lr1e-3_1x-duc.yaml --checkpoint  pretrained_models/fast_421_res152_256x192.pth --video ../dataset/NonViolence/NV_1_frames/frame0.jpg --outdir ../dataset --sp')