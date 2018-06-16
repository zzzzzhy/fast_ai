import face_test as face_model
import argparse
import cv2
import sys
import numpy as np

import time

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
orig_img = cv2.imread('Tom_Hanks_54745.png')

start = time.time()
img = model.get_input(orig_img)
done = time.time()
print('first face detection {}'.format(done - start))

start = time.time()
for i in range (1,1000):
    step_start = time.time()
    img = model.get_input(orig_img)
    step_done = time.time()
    print('step {} face detection {}'.format(i,(step_done - step_start)))

done = time.time()
print('face detection everage {}'.format((done - start)/1000))

