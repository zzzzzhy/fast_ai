import face_model
#import argparse
#import sys
#import time
#import cv2

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
img = cv2.imread('Tom_Hanks_54745.png')

start = time.time()
img = model.get_input(img)
done = time.time()
print('face detection {}'.format(done - start))

start = time.time()
f1 = model.get_feature(img)
done = time.time()
print('face recognition {}'.format(done - start))

start = time.time()
for i in range (1,100):
    f1 = model.get_feature(img)

done = time.time()
print('face recognition everage {}'.format((done - start)/100))


