from tkinter import Frame
import cv2
from face_recognition_models import face_recognition_model_location
import numpy as np
import face_recognition
import os
from datetime import datetime
import time
import re

unknownPrompt = 'Unknown face - click \'y\' to add to database'
typeNamePrompt = 'Type person\'s name and press \'1\' to accept'

shape_model = "large"
detector_model = "hog"
pictures_path = 'pics'
attendance_path = 'attendanceLists'
images = []
databaseNames = []
resizeFactor = 3
totalFrames = 0
totalTime = 0
lostFramesCount = 0

fileNamesList = os.listdir(pictures_path)

detectedFramesCount = 0
detectedFramesThreshold = 5

for fileName in fileNamesList:
    currentImg = cv2.imread(f'{pictures_path}/{fileName}')
    images.append(currentImg)
    databaseNames.append(os.path.splitext(fileName)[0])

def findEncodings(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        encodedImg = face_recognition.face_encodings(img, model=shape_model)[0]
        encodedList.append(encodedImg)
    return encodedList

def getImageEncoding(imgName):
    img = cv2.imread(f'{pictures_path}/{imgName}.jpg')
    images.append(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    encodedImg = face_recognition.face_encodings(img, model=shape_model)[0]
    return encodedImg

def markPresent(name):
    now = datetime.now()
    str_timestamp = date_time = now.strftime("%d_%m_%Y__%H")
    with open(f'./{attendance_path}/Attendance_{str_timestamp}.csv', 'a+') as f:
        if os.stat(f'{attendance_path}/Attendance_{str_timestamp}.csv').st_size == 0:
            f.write('Name,Time')

        f.seek(0)
        myDataList = f.readlines()

        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        
        # print(nameList)
        if name not in nameList:
            now = datetime.now()
            time_str = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time_str}')

encodedListOfFaces = findEncodings(images)

videoCapture = cv2.VideoCapture(0)

while True:
    # if totalFrames > 30:
    #     totalFrames = 0
    #     totalTime = 0

    # totalFrames += 1

    start = time.time()
    success, img = videoCapture.read()
    imgSmall = cv2.resize(img, (0,0), None, 1/resizeFactor, 1/resizeFactor)
    faceLocationsInFrame = face_recognition.face_locations(imgSmall, model=detector_model)
    currentFrameEncodings = face_recognition.face_encodings(imgSmall, faceLocationsInFrame, model=shape_model)
    
    imgMidHorizontal = int(img.shape[0] / 2)
    imgMidVertical = int(img.shape[1] / 2)

    if not currentFrameEncodings:
        detectedFramesCount = 0
        lostFramesCount += 1
    for encoding, location in zip(currentFrameEncodings, faceLocationsInFrame):
        matches = face_recognition.compare_faces(encodedListOfFaces, encoding)
        faceDistance = face_recognition.face_distance(encodedListOfFaces, encoding)
        top, right, bottom, left = location
        faceMatchIndex = np.argmin(faceDistance)


        if matches[faceMatchIndex]:
            detectedFramesCount += 1
            cv2.rectangle(img, (left * resizeFactor, top * resizeFactor), (right * resizeFactor, bottom * resizeFactor), (0, 255 ,0), 3)
            matchName = databaseNames[faceMatchIndex]
            cv2.putText(img, matchName, (left * resizeFactor, top * resizeFactor - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255 ,0))
            if detectedFramesCount > detectedFramesThreshold:
                markPresent(matchName)
        else:
            detectedFramesCount = 0
            key = cv2.waitKey(1) & 0xFF
            newName = ''
            inputChar = ''
            if key == ord('y'):
                success, cleanImg = videoCapture.read()
                while (inputChar != '1'):
                    inputChar = chr(cv2.waitKey(1) & 0xFF)
                    if  re.match("[a-z]", inputChar) or \
                        re.match("[A-Z]", inputChar) or \
                        re.match("[ ]", inputChar):
                        myTuple = (newName, inputChar)
                        newName = ''.join(myTuple)
                    
                    cv2.putText(img, typeNamePrompt, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0 , 255))
                    cv2.putText(img, "NEW NAME:", (imgMidHorizontal - int(imgMidHorizontal / 4), imgMidVertical - 20), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255 , 0))
                    cv2.putText(img, "{}".format(newName), (imgMidHorizontal - int(imgMidHorizontal / 4), imgMidVertical + 20), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255 , 0))
                    cv2.imshow('WEBCAM', img)

                cv2.imwrite('{}/{}.jpg'.format(pictures_path, newName), cleanImg)
                if newName not in databaseNames:
                    databaseNames.append(newName)
                    encodedListOfFaces.append(getImageEncoding(newName))
                    print(f'LIST OF NAMES\n{databaseNames}\n')

            cv2.putText(img, unknownPrompt, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0 , 255))
            cv2.rectangle(img, (left * resizeFactor, top * resizeFactor), (right * resizeFactor, bottom * resizeFactor), (255, 0 ,0), 3)

    cv2.imshow('WEBCAM', img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:   
        break  # esc to quit

    end = time.time()
    seconds = end - start

    # totalTime += seconds
    fps = 1 / seconds 
    print("Frames per second: {}".format(fps))
cv2.destroyAllWindows()