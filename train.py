import itertools

from extract import extract_dataset_archive
from split import convert_train_set_to_images
import os
import keras
from keras.layers import Dense, LSTM, Activation
from keras.models import Sequential
import pandas as pd
import numpy as np

# os.system('python scripts/demo_inference.py --cfg pretrained_models/256x192_res152_lr1e-3_1x-duc.yaml --checkpoint  pretrained_models/fast_421_res152_256x192.pth --video ../dataset/NonViolence/NV_1.mp4 --outdir ../dataset --sp')

def prepare_dataset():
    extract_dataset_archive()
    convert_train_set_to_images()


def convert_videos_to_json(path_in, path_out):
    os.system('python scripts/demo_inference.py --cfg pretrained_models/256x192_res152_lr1e-3_1x-duc.yaml '
              '--checkpoint  pretrained_models/fast_421_res152_256x192.pth --video ' + path_in + ' --outdir ' + path_out + ' --sp')


#prepare_dataset()
#os.chdir("AlphaPose")

# for i in range(1, 1001):
#    file_in = str("../dataset/Violence/V_") + str(i) + ".mp4"
#    file_out = str("../dataset/results/V_") + str(i) + "_json"
#    convert_videos_to_json(file_in, file_out)

#for i in range(1, 1001):
#    file_in = str("../dataset/NonViolence/NV_") + str(i) + ".mp4"
#    file_out = str("../dataset/results/NV_") + str(i) + "_json"
#    convert_videos_to_json(file_in, file_out)

import keras
# from keras.layers import Sequential, Dense, LSTM, Activation
import pandas as pd
import numpy as np

person_count = 5

data=pd.read_csv("dataset/csvs/NV_100_json.csv").to_numpy()

val_cumulative = {}
frame_ids = set()
for person in data:
    frame_ids.add(person[0])
    frame, person_id = person[:2]

    vel_array = []
    for vel in range(5, 70, 4):
        vel_array.append(vel)

    total_vel = sum(vel_array)

    if person_id in val_cumulative:
        val_cumulative[person_id] += total_vel
    else:
        val_cumulative[person_id] = 0

val_cumulative = {k: v for k, v in sorted(val_cumulative.items(), key=lambda item: item[1], reverse=True)}
top_people = dict(itertools.islice(val_cumulative.items(), person_count))
top_people_dataset = dict()
top_people.keys()

for frame_id in frame_ids:
    top_people_dataset[frame_id] = []
    people = [item for item in data if item[0] == frame_id]
    for person in people:
        if person[1] in top_people:
            top_people_dataset[person[0]].append(person)



model = Sequential()
model.add(LSTM(100, input_shape=(150, 2048))) # 150, 2048 = n_chunks, chunk_size
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.train_on_batch()


x_train = []
y_train = []
epochs = 1500
batchs = 100
for i in top_people_dataset:
    x_train.append(i[0])
    y_train.append(np.array(i[1]))

model.fit(x_train, y_train, epochs, batch_size=batchs, verbose=1)
model.save("rnn.h5", overwrite=True)