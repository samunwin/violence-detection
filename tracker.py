from pyimagesearch.centroidtracker import CentroidTracker
import json
import sys
import numpy as np
import csv
import os 
from scipy.spatial import distance as dist
from math import sqrt

results_dir = "dataset/results/"
csv_dir = "dataset/csvs/"

file_count = 0

# os.makedirs(csv_dir)

for dir in os.listdir(results_dir):
    if file_count == 1:
            break
    for file in os.listdir(results_dir + dir):

        ct = CentroidTracker()

        if dir.startswith("NV"):
            violent = 0
        else:
            violent = "todo"

        with open(results_dir + dir + "/"+ file, 'r') as file:
            json_src = json.loads(file.read())

        image_ids = set()

        for jsonobj in json_src:
            frame = jsonobj['image_id']
            image_ids.add(frame)

        with open(csv_dir + dir + '.csv', mode='w', newline='') as csv_file:
            fieldnames = ['frame', 'object_id', 'keypoints', 'velocities','violent']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            frame_num = 0
            previous_val = {}

            for image_id in image_ids:
                results = [item for item in json_src if item['image_id'] == image_id]

                rects = []
                result_count = 0

                for result in results:

                    minx = sys.maxsize
                    miny = sys.maxsize
                    maxx = 0
                    maxy = 0

                    xvals = []
                    yvals = []
                    pose_centroids = []

                    for i in range(0, 51, 3):
                        x = result['keypoints'][i]
                        xvals.append(x)
                        if x < minx:
                            minx = x
                        if x > maxx:
                            maxx = x

                    for i in range(1, 51, 3):
                        y = result['keypoints'][i]
                        yvals.append(y)
                        if y < miny:
                            miny = y
                        if y > maxy:
                            maxy = y

                    for i in range(0, 17):
                        pose_centroids.append([xvals[i], yvals[i]])
                    
                    cX = int((minx + maxx) / 2.0)
                    cY = int((miny + maxy) / 2.0)

                    D = [0 for i in range(0,17)]

                    vels = [0 for i in range(0,17)]

                    if previous_val != {}:
                        vels = []
                        D = dist.cdist(np.array(previous_val['centroids']), np.array(pose_centroids))
                        row = D.min(axis=1).argsort()
                        col = D.argmin(axis=1)[row]
                        
                        for i in row:
                            if row[i] == 0 and col[i] == 0:
                                vels.append(0)
                            else:
                                vels.append(1 / sqrt(row[i]**2 + col[i]**2))

                    previous_val = {'centroids': pose_centroids}

                    rects.append([minx, miny, maxx, maxy])

                    objects = ct.update(rects)

                    print(image_id, " Result: ", result_count)

                    for (objectID, centroid) in objects.items():
                        if np.array_equal(centroid, [cX, cY]):  
                            writer.writerow({'frame': frame_num, 'object_id': objectID, 'keypoints': result['keypoints'], 'velocities': vels,'violent': violent})
                            print("ID:", format(objectID), "Centroid:", format(centroid), "Pose Info: ", 'velocities:', vels, format(result['keypoints']))
                    result_count += 1
                frame_num += 1
    file_count += 1