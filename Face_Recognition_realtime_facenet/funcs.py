# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 08:10:46 2020

@author: NITHIN BURRA
"""

import pickle
import numpy as np
#import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import euclidean

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

#load_picle
def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict