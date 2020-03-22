from pyimagesearch.centroidtracker import CentroidTracker
import json
import sys
import numpy as np
import csv
import os 
from scipy.spatial import distance as dist
from math import sqrt

from util import get_pose_part

results_dir = "dataset/results/"
csv_dir = "dataset/csvs/"

file_count = 0

# os.makedirs(csv_dir)

for dir in os.listdir(results_dir):
    if file_count == 10:
        break

    for file in os.listdir(results_dir + dir):

        ct = CentroidTracker()

        if dir.startswith("NV"):
            violent = 0
        else:
            violent = 1

        with open(results_dir + dir + "/"+ file, 'r') as file:
            json_src = json.loads(file.read())

        image_ids = set()

        for jsonobj in json_src:
            frame = jsonobj['image_id']
            image_ids.add(frame)

        with open(csv_dir + dir + '.csv', mode='w', newline='') as csv_file:
            fieldnames = ['frame', 'object_id']
            for i in range(0, 17):
                fieldnames.append(get_pose_part(i)+"X")
                fieldnames.append(get_pose_part(i)+"Y")
                fieldnames.append(get_pose_part(i)+"C")
                fieldnames.append(get_pose_part(i) + "V")

            fieldnames.append('violent')
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

                    cX = int((minx + maxx) / 2.0)
                    cY = int((miny + maxy) / 2.0)

                    xvals = list(map(lambda value: value - cX, xvals))
                    yvals = list(map(lambda value: value - cY, yvals))

                    for i in range(0, 17):
                        pose_centroids.append([xvals[i], yvals[i]])

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
                                vels.append(sqrt(row[i]**2 + col[i]**2))

                    previous_val = {'centroids': pose_centroids}

                    rects.append([minx, miny, maxx, maxy])

                    objects = ct.update(rects)

                    print(image_id, " Result: ", result_count)

                    for (objectID, centroid) in objects.items():
                        if np.array_equal(centroid, [cX, cY]):
                            row = {'frame': frame_num, 'object_id': objectID, 'violent': violent}

                            for i in range(0, 51, 3):
                                label = get_pose_part(i//3)+"X"
                                row[label] = result['keypoints'][i]

                            for i in range(1, 51, 3):
                                label = get_pose_part(i//3)+"Y"
                                row[label] = result['keypoints'][i]

                            for i in range(2, 51, 3):
                                label = get_pose_part(i//3)+"C"
                                row[label] = result['keypoints'][i]

                            for i in range(0, 17):
                                value = vels[i]
                                label = get_pose_part(i)+"V"
                                row[label] = value
                            #print(row)
                            writer.writerow(row)
                            #print("ID:", format(objectID), "Centroid:", format(centroid), "Pose Info: ", 'velocities:', vels, format(result['keypoints']))
                    result_count += 1
                frame_num += 1
    file_count += 1
