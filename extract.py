import os
import shutil
import zipfile


def extract_dataset_archive():
    print("Locating dataset...")
    current_dir = os.path.basename(os.path.realpath(''))

    if current_dir != 'violence-detection':
        raise ValueError("Execution outside violence-detection root folder")

    if not os.path.exists('dataset/Violence') or not os.path.exists('dataset/NonViolence'):
        if not os.path.exists('dataset/real-life-violence-situations-dataset.zip'):
            raise ValueError("Dataset not found download the Kaggle real-life-violence-situations-dataset.zip from "
                             "https://www.kaggle.com/mohamedmustafa/real-life-violence-situations-dataset/data and "
                             "place the zip file in violence-detection/dataset")

        print("Extracting dataset...")
        with zipfile.ZipFile('dataset/real-life-violence-situations-dataset.zip', 'r') as zip_ref:
            zip_ref.extractall('dataset')

        shutil.copytree('dataset/Real Life Violence Dataset/Violence', 'dataset/Violence')
        shutil.copytree('dataset/Real Life Violence Dataset/NonViolence', 'dataset/NonViolence')
        print('Cleaning up archive...')
        shutil.rmtree('dataset/real life violence situations')
        shutil.rmtree('dataset/Real Life Violence Dataset')
        os.remove('dataset/real-life-violence-situations-dataset.zip')
