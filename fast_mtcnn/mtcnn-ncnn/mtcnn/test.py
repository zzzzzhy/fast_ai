import face_detection as m

for i in range(1000):
  result = m.detect('./1_854x480.jpg')
print(result)
