# Violence-detection
A project using a violence/non violence dataset and the AlphaPose project tool and Kalman filtering techniques
to create a violence detection model for video sequences. 

## Training Setup Guide
1. Download the real-life-violence-situations-dataset.zip from 
https://www.kaggle.com/mohamedmustafa/real-life-violence-situations-dataset/data 
and place the zip file in violence-detection/dataset

2. Clone the AlphaPose Git repository from
https://github.com/MVIG-SJTU/AlphaPose.git
into the parent folder of violence detection project root.

3. Have Anaconda installed or otherwise have Conda active on the OS console.

    Conda: https://pypi.org/project/conda/ or
    
    Anaconda https://docs.anaconda.com/anaconda/install/
    
4. Create conda environment (run in project root folder):
 
conda create --name <env> --file requirements.txt

5. Run Alphapose's setup.py in the Alphapose root folder
