from pyimagesearch.centroidtracker import CentroidTracker
import json
import sys
import numpy as np
import csv
import os 

ct = CentroidTracker()

results_dir = "dataset/results/"
csv_dir = "dataset/csvs/"

os.makedirs(csv_dir)

for dir in os.listdir(results_dir):
    for file in os.listdir(results_dir + dir):

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
            fieldnames = ['frame', 'object_id', 'keypoints', 'violent']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            frame_num = 0
            for image_id in image_ids:
                results = [item for item in json_src if item['image_id'] == image_id]

                rects = []
                result_count = 0

                for result in results:

                    minx = sys.maxsize
                    miny = sys.maxsize
                    maxx = 0
                    maxy = 0

                    for i in range(0, 51, 3):
                        x = result['keypoints'][i]
                        if x < minx:
                            minx = x
                        if x > maxx:
                            maxx = x

                    for i in range(1, 51, 3):
                        y = result['keypoints'][i]
                        if y < miny:
                            miny = y
                        if y > maxy:
                            maxy = y
                    
                    cX = int((minx + maxx) / 2.0)
                    cY = int((miny + maxy) / 2.0)

                    rects.append([minx, miny, maxx, maxy])

                    objects = ct.update(rects)

                    #print(image_id, " Result: ", result_count)

                    for (objectID, centroid) in objects.items():
                        if np.array_equal(centroid, [cX, cY]):  
                            writer.writerow({'frame': frame_num, 'object_id': objectID, 'keypoints': result['keypoints'], 'violent': violent})
                            #print("ID:", format(objectID), "Centroid:", format(centroid), "Pose Info: ", format(result['keypoints']))
                    result_count += 1
                frame_num += 1