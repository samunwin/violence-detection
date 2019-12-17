import cv2
from os import makedirs

#Edit the path to the data set
video_path = "/Users/szymonskrzek/Documents/real-life-violence-situations-dataset/real-life-violence-dataset/"

def convertToFramesAndSave(path, amount):
    for i in range(1, amount):
        file = str(path) + str(i) + ".mp4"
        vidcap = cv2.VideoCapture(file)
        success, image = vidcap.read()
        count = 0
        makedirs(str(path) + str(i) + "_frames")
        while success:
            cv2.imwrite(str(path) + str(i) + "_frames/frame%d.jpg" % count, image)  # save frame as JPEG file
            success, image = vidcap.read()
            print('Read a new frame: ', success)
            print('frame saved under '+ str(path) + str(i) + "/frames/frame%d.jpg" % count)
            count += 1


convertToFramesAndSave((video_path + "NonViolence/NV_"), 1001)
convertToFramesAndSave((video_path + "Violence/V_"), 1001)