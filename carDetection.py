import os
import re
import cv2 # opencv library
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

#frames all in one
col_Frames = os.listdir('frames/')

#sort the file names
col_Frames.sort(key=lambda f: int(re.sub('\D', '', f)))

#empty list for frames
allFrames=[]

for i in col_Frames:
	#read frames
	img = cv2.imread('frames/'+i)
	#add frames to the list
	allFrames.append(img)

# kernel for image dilation
kernel = np.ones((4,4),np.uint8)

# font style
font = cv2.FONT_HERSHEY_SIMPLEX

# directory to save the ouput frames
pathIn = "contour_frames/"

for i in range(len(allFrames)-1):
    
    # frame differencing
    grayA = cv2.cvtColor(allFrames[i], cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(allFrames[i+1], cv2.COLOR_BGR2GRAY)
    sep_image = cv2.absdiff(grayB, grayA)
    
    # image thresholding
    ret, thresh = cv2.threshold(sep_image, 30, 255, cv2.THRESH_BINARY)
    
    # image dilation
    dilated = cv2.dilate(thresh,kernel,iterations = 1)
    
    # find contours
    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    # shortlist contours appearing in the detection zone
    valid_contours = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if (x <= 200) & (y >= 80) & (cv2.contourArea(contour) >= 25):
            if (y >= 90) & (cv2.contourArea(contour) < 40):
                break
            valid_contours.append(contour)

    len(valid_contours)
            
    # add contours to original frames
    dmy = allFrames[i].copy()
    cv2.drawContours(dmy, valid_contours, -1, (0,0,200), 2)
    
    cv2.putText(dmy, "Vehicles Detected: " + str(len(valid_contours)), (55, 15), font, 0.6, (0, 0, 0), 2)
    cv2.putText(dmy, "Detection Zone", (35, 90), font, 0.4, (0, 0, 0), 1)
    cv2.line(dmy, (0, 80),(256,80),(255, 255, 255))
    cv2.imwrite(pathIn+str(i)+'.png',dmy)

# video output
pathOut = 'vehicle_detection.mp4'
fps = 14.0

array_of_frames = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

files.sort(key=lambda f: int(re.sub('\D', '', f)))

for i in range(len(files)):
    filename=pathIn + files[i]
    
    #read frames
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)

    #inserting the frames into an image array
    array_of_frames.append(img)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(array_of_frames)):
    # writing to a image array
    out.write(array_of_frames[i])

out.release()