import cv2
import numpy as np
from pose import Pose as m

width = 513
height = 513

d = m()
d.init("model/posenet.pb")

def read_imgfile(path, width, height):
    img = cv2.imread(path)
    img = cv2.resize(img, (width,height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    img = img * (2.0 / 255.0) - 1.0
    return img

input_image = read_imgfile("./images/tennis_in_crowd.jpg",width,height)
input_image = np.array(input_image,dtype=np.float32)
input_image = input_image.reshape(1,width,height,3)

heatmap = np.ndarray(shape=(1,33,33,17), dtype=np.float32)
offset_2 = np.ndarray(shape=(1,33,33,34), dtype=np.float32)
displacement_fwd_2 = np.ndarray(shape=(1,33,33,32), dtype=np.float32)
displacement_bwd_2 = np.ndarray(shape=(1,33,33,32), dtype=np.float32)

print(input_image.shape)
print(">>> test: ")
ret = d.detect(input_image,heatmap,offset_2,displacement_fwd_2,displacement_bwd_2)

print("========")
print("heatmaps")
heatmaps_result = heatmap[0]
print(heatmaps_result[0:1, 0:1, :])
print(heatmaps_result.shape)
print(np.mean(heatmaps_result))

print(heatmap)
#print(offset_2)
print(">>> test again: ")
ret = d.detect(input_image,heatmap,offset_2,displacement_fwd_2,displacement_bwd_2)
