# Violence-detection
A project using a violence/non violence dataset and the AlphaPose project tool and Kalman filtering techniques
to create a violence detection model for video sequences. 

## Training Setup Guide
1. Download the real-life-violence-situations-dataset.zip from 
https://www.kaggle.com/mohamedmustafa/real-life-violence-situations-dataset/data 
and place the zip file in violence-detection/dataset

2. (Optional) Download the .zip's for the JSON and CSVs and unzip to make project structure look like:

```
violence-detection
│   README.md
│   ...
│
└───dataset/
    │  
    └───csvs/
        │   NV_1_json.csv
        │   ...
        │
        results/$$
        └───NV_1_json/
        │   └───alphapose-results.json
        │   ...
```

Download links are [here](https://drive.google.com/open?id=1hRBHpE1ead11aoiUjq2fhJ-WbYBuFYJH) and [here](https://drive.google.com/open?id=1weHgrOs7svr-p4vxomRJiqY6w_V6ImyM)

3. Clone the AlphaPose Git repository from
https://github.com/MVIG-SJTU/AlphaPose.git
into the parent folder of violence detection project root.

4. Have Anaconda installed or otherwise have Conda active on the OS console.

    Conda: https://pypi.org/project/conda/ or
    
    Anaconda https://docs.anaconda.com/anaconda/install/
    
5. Create conda environment (run in project root folder):
`conda create --name <env> --file requirements.txt`

6. Run Alphapose's setup.py in the Alphapose root folder:
`python setup.py build develop`