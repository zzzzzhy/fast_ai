import face_detection as m

m.set_minsize(40)
m.set_threshold(0.6,0.7,0.8)
m.set_num_threads(2)

for i in range(1000):
  result = m.detect('./1_854x480.jpg')
print(result)
