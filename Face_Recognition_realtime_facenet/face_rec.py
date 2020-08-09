# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:51:50 2020

@author: NITHIN BURRA
"""

import os
import pickle
import numpy as np
import cv2
import mtcnn
from keras.models import load_model
from scipy.spatial.distance import cosine
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import euclidean
from datetime import datetime

#mark attendance
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%d/%m/%Y,%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
# get encode
def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

#get_face
def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


#normalize
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

l2_normalizer = Normalizer('l2')

#align
def align(img, left_eye_pos, right_eye_pos, size=(112, 112), eye_pos=(0.35, 0.4)):
    width, height = size
    eye_pos_w, eye_pos_h = eye_pos

    l_e, r_e = left_eye_pos, right_eye_pos

    dy = r_e[1] - l_e[1]
    dx = r_e[0] - l_e[0]
    dist = euclidean(l_e, r_e)
    scale = (width * (1 - 2 * eye_pos_w)) / dist

    # get rotation
    center = ((l_e[0] + r_e[0]) // 2, (l_e[1] + r_e[1]) // 2)
    angle = np.degrees(np.arctan2(dy, dx)) + 360

    m = cv2.getRotationMatrix2D(center, angle, scale)
    tx = width * 0.5
    ty = height * eye_pos_h
    m[0, 2] += (tx - center[0])
    m[1, 2] += (ty - center[1])

    aligned_face = cv2.warpAffine(img, m, (width, height))
    return aligned_face
print("*********************************************************")
encoder_model = input('Path for model: ')
people_dir = input('Path for dataset: ')

while True:
    print("0: To save encodings to pickle file press 0")
    print("1: To continue press 1")
    choice = input("Enter your choice: ")
    choice = int(choice)
    if choice in [0,1]:
        break
if choice == 0:
    print("Don't forget to give .pkl file extension")
    encodings_path = input('Path for encodings: ')
elif choice == 1:
    print("Encodings will not be saved")
print("********************************************************")
    

required_size = (160,160)

face_detector = mtcnn.MTCNN()
if os.path.exists(encoder_model):
    face_encoder = load_model(encoder_model)

encoding_dict = dict()

for person_name in os.listdir(people_dir):
    person_dir = os.path.join(people_dir, person_name)
    encodes = []
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detector.detect_faces(img_rgb)
        if results:
            box = max(results, key=lambda b: b['box'][2] * b['box'][3])
            l_e = box['keypoints']['left_eye']
            r_e = box['keypoints']['right_eye']
            face = align(img, l_e, r_e, size=required_size, eye_pos=(0.35, 0.4))
            face = normalize(face)
            encode = get_encode(face_encoder, face, required_size)
            encodes.append(encode)
    if encodes:
        encode = np.sum(encodes, axis=0)
        encode = l2_normalizer.transform(encode.reshape(1, -1))
        encoding_dict[person_name] = encode[0]
for key in encoding_dict.keys():
    print(key)

if encodings_path:
    with open(encodings_path, 'bw') as file:
        pickle.dump(encoding_dict, file)
    print("Encodings Successfully saved")
   
def recognize(img,
              detector,
              encoder,
              encoding_dict,
              recognition_t=0.5,
              confidence_t=0.99,
              required_size=(160, 160), ):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        l_e = res['keypoints']['left_eye']
        r_e = res['keypoints']['right_eye']
        face = align(img_rgb, l_e, r_e, required_size)
        _, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist
                
        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 3)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 3)
            cv2.putText(img, name + f'_{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
            markAttendance(name)
            
    return img


vc = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter('output.avi', fourcc, 10.0, (640, 480))
while vc.isOpened():
    ret, frame = vc.read()
    if not ret:
        print("no frame:")
        break
    frame = recognize(frame, face_detector, face_encoder, encoding_dict)
    cv2.imshow('Face Recognition', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Successfully completed")
        break
vc.release()
out.release()
cv2.destroyAllWindows()
