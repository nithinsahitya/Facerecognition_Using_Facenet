# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:34:14 2020
"""

import os
import pickle
import numpy as np
import cv2
import mtcnn
from keras.models import load_model
from funcs import normalize, get_encode, l2_normalizer,align

# hyper-parameters
encoder_model = 'facenet_keras.h5'
people_dir = 'dataset/train'
encodings_path = 'encodings.pkl'
required_size = (160, 160)

face_detector = mtcnn.MTCNN()
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

with open(encodings_path, 'bw') as file:
    pickle.dump(encoding_dict, file)
