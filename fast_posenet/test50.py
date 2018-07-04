import cv2
import numpy as np
import time
from pose import Pose as m

width = 513
height = 513

d = m()
d.init("./pypose/model/posenet_50.pb")

def read_imgfile(path, width, height):
    img = cv2.imread(path)
    img = cv2.resize(img, (width,height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    img = img * (2.0 / 255.0) - 1.0
    return img

heatmap = np.ndarray(shape=(1,33,33,17), dtype=np.float32)
offset_2 = np.ndarray(shape=(1,33,33,34), dtype=np.float32)
displacement_fwd_2 = np.ndarray(shape=(1,33,33,32), dtype=np.float32)
displacement_bwd_2 = np.ndarray(shape=(1,33,33,32), dtype=np.float32)

start = time.time()
input_image = read_imgfile("./pypose/images/tennis_in_crowd.jpg",width,height)
input_image = np.array(input_image,dtype=np.float32)
input_image = input_image.reshape(1,width,height,3)

end = time.time()
print("image loading took: {}".format(end-start))

ret = d.detect(input_image,heatmap,offset_2,displacement_fwd_2,displacement_bwd_2)

start = time.time()
for i in range(1000):
    step_start = time.time()
    input_image = read_imgfile("./pypose/images/tennis_in_crowd.jpg",width,height)
    input_image = np.array(input_image,dtype=np.float32)
    input_image = input_image.reshape(1,width,height,3)

    ret = d.detect(input_image,heatmap,offset_2,displacement_fwd_2,displacement_bwd_2)
    step_end = time.time()
    print("step {} took: {}".format( i,step_end-step_start))
end = time.time()
print("Everage : {}".format((end-start)/1000))
