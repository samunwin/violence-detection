from extract import extract_dataset_archive
from split import convert_train_set_to_images


def prepare_dataset():
    extract_dataset_archive()
    convert_train_set_to_images()