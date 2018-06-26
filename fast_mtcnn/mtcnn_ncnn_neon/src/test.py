import face_detection as m
m.init('../models/ncnn/')
for i in range(1000):
  result = m.detect('../images_480p/1_854x480.jpg')
print(result)
