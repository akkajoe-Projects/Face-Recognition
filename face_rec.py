import cv2 as cv
import numpy as np
import face_recognition

img = face_recognition.load_image_file('Anni.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
face_loc = face_recognition.face_locations(img)[0]
encodeimg = face_recognition.face_encodings(img)[0]
cv.rectangle(img, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (124,252,0), 2)
name = 'Maggot'
# cv.imshow('akka', img)
# cv.waitKey(0)

vid_cap = cv.VideoCapture(0)
while True:
    success, image = vid_cap.read()
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    faces_frame= face_recognition.face_locations(image)
    encodes_frame = face_recognition.face_encodings(image, faces_frame)
    # cv.rectangle(img, (faces_frame[3], faces_frame[0]), (faces_frame[1], faces_frame[2]), (124,252,0), 2)
    for encode_face, faceloc in zip(encodes_frame, faces_frame):
        matches = face_recognition.compare_faces([encodeimg], encode_face)
        face_dis = face_recognition.face_distance([encodeimg], encode_face)
        print(matches, face_dis)
        if matches == [True]:
            print('inside matches')
            print(name)
            y1,x2,y2,x1 = faceloc
            cv.rectangle(image, (x1,y1), (x2,y2), (124,252,0), 2)
            cv.rectangle(image, (x1, y2-35), (x2, y2), cv.FILLED)
            cv.putText(image, name, (x1+6, y2-6), cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv.imshow('Webcam',image)
    cv.waitKey(1)