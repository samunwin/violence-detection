from extract import extract_dataset_archive
from split import convert_train_set_to_images
import os


# os.system('python scripts/demo_inference.py --cfg pretrained_models/256x192_res152_lr1e-3_1x-duc.yaml --checkpoint  pretrained_models/fast_421_res152_256x192.pth --video ../dataset/NonViolence/NV_1.mp4 --outdir ../dataset --sp')

def prepare_dataset():
    extract_dataset_archive()
    convert_train_set_to_images()


def convert_videos_to_json(path_in, path_out):
    os.system('python scripts/demo_inference.py --cfg pretrained_models/256x192_res152_lr1e-3_1x-duc.yaml '
              '--checkpoint  pretrained_models/fast_421_res152_256x192.pth --video ' + path_in + ' --outdir ' + path_out + ' --sp')


prepare_dataset()
os.chdir("AlphaPose")

for i in range(1, 1001):
    file_in = str("../dataset/Violence/V_") + str(i) + ".mp4"
    file_out = str("../dataset/results/V_") + str(i) + "_json"
    convert_videos_to_json(file_in, file_out)

for i in range(1, 1001):
    file_in = str("../dataset/NonViolence/NV_") + str(i) + ".mp4"
    file_out = str("../dataset/results/NV_") + str(i) + "_json"
    convert_videos_to_json(file_in, file_out)

