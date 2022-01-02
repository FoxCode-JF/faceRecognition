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

pictures_path = 'pics'
attendance_path = 'attendanceLists'
nameList = []
images = []
databaseNames = []
resizeFactor = 4

fileNamesList = os.listdir(pictures_path)

# print(fileNamesList)

for fileName in fileNamesList:
    currentImg = cv2.imread(f'{pictures_path}/{fileName}')
    images.append(currentImg)
    databaseNames.append(os.path.splitext(fileName)[0])

def findEncodings(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodedImg = face_recognition.face_encodings(img)[0]
        encodedList.append(encodedImg)
    return encodedList

def markPresent(name, nameList):
    now = datetime.now()
    str_timestamp = date_time = now.strftime("%d_%m_%Y__%H")
    with open(f'./{attendance_path}/Attendance_{str_timestamp}.csv', 'a+') as f:
        if os.stat(f'{attendance_path}/Attendance_{str_timestamp}.csv').st_size == 0:
            f.write('Name,Time')

        f.seek(0)
        myDataList = f.readlines()
        # print(myDataList)
        
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        
        # print(nameList)
        if name not in nameList:
            now = datetime.now()
            time_str = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time_str}')



encodedListOfFaces = findEncodings(images)
print(len(encodedListOfFaces))

videoCapture = cv2.VideoCapture(0)

while True:
    success, img = videoCapture.read()
    imgSmall = cv2.resize(img, (0,0), None, 1/resizeFactor, 1/resizeFactor)
    faceLocationsInFrame = face_recognition.face_locations(imgSmall)
    currentFrameEncodings = face_recognition.face_encodings(imgSmall, faceLocationsInFrame)
    
    imgMidHorizontal = int(img.shape[0] / 2)
    imgMidVertical = int(img.shape[1] / 2)

    for encoding, location in zip(currentFrameEncodings, faceLocationsInFrame):
        matches = face_recognition.compare_faces(encodedListOfFaces, encoding)
        faceDistance = face_recognition.face_distance(encodedListOfFaces, encoding)
        top, right, bottom, left = location
        faceMatchIndex = np.argmin(faceDistance)

        if matches[faceMatchIndex]:
            cv2.rectangle(img, (left * resizeFactor, top * resizeFactor), (right * resizeFactor, bottom * resizeFactor), (0, 255 ,0), 3)
            matchName = databaseNames[faceMatchIndex].upper()
            cv2.putText(img, matchName, (left * resizeFactor, top * resizeFactor - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255 ,0))
            # print(matchName)
            markPresent(matchName, nameList=nameList)
        else:
            key = cv2.waitKey(1) & 0xFF
            newName = ''
            inputChar = ''
            if key == ord('y'):
                success, cleanImg = videoCapture.read()
                # success, img = videoCapture.read()
                while (inputChar != '1'):
                    inputChar = chr(cv2.waitKey(1) & 0xFF)
                    if re.match("[a-z]", inputChar):
                        print(inputChar)
                        myTuple = (newName, inputChar)
                        newName = ''.join(myTuple)
                    # print("width : {}, height : {}".format(imgMidHorizontal, imgMidVertical))
                    cv2.putText(img, typeNamePrompt, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0 , 255))
                    cv2.putText(img, "NEW NAME: {}".format(newName), (imgMidHorizontal, imgMidVertical), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0 , 0))
                    cv2.imshow('WEBCAM', img)
                # print(newName)
                cv2.imwrite('{}/{}.jpg'.format(pictures_path, newName), cleanImg)
                nameList.append(newName)

                # putText(img, unknownPrompt, (left * resizeFactor, top * resizeFactor - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255 ,0))
            cv2.putText(img, unknownPrompt, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0 , 255))
            cv2.rectangle(img, (left * resizeFactor, top * resizeFactor), (right * resizeFactor, bottom * resizeFactor), (255, 0 ,0), 3)

    cv2.imshow('WEBCAM', img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:   
        break  # esc to quit
cv2.destroyAllWindows()