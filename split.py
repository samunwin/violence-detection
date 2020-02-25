import os
import time

import cv2

last_times = []


def convert_train_set_to_images():
    if not os.path.exists('dataset/NonViolence/NV_1_frames') and not os.path.exists('dataset/Violence/V_1_frames'):
        print("Videos need converting")
        print("Converting Non Violence")
        convert_to_frames_and_save("dataset/NonViolence/NV_", 1001)
        print("Converting Violence")
        convert_to_frames_and_save("dataset/Violence/V_", 1001)
    else:
        print("Conversion not required")


def convert_to_frames_and_save(path, amount):
    last_t = 0
    for i in range(1, amount):
        t = time.time()
        file = str(path) + str(i) + ".mp4"
        vidcap = cv2.VideoCapture(file)
        success, image = vidcap.read()
        count = 0
        os.makedirs(str(path) + str(i) + "_frames")
        while success:
            cv2.imwrite(str(path) + str(i) + "_frames/frame%d.jpg" % count, image)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
        last_t = time.time() - t
        print(get_remaining_time(i, amount, last_t), end="\r")


def get_remaining_time(i, total, time):
    last_times.append(time)
    len_last_t = len(last_times)
    if len_last_t > 5:
        last_times.pop(0)
    mean_t = sum(last_times) // len_last_t
    remain_s_tot = mean_t * (total - i + 1)
    remain_m = remain_s_tot // 60
    remain_s = remain_s_tot % 60
    loading = ['/','-','\\','|']
    return loading[i % 4]+" " + f"{remain_m}m{remain_s}s"
