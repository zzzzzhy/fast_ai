import face_detection as m
import json
m.load('../models')
result =m.detect('test.jpg','./output')
#print(result)

obj = json.loads(result)

print(obj['result'][0]['score'])
#m.test()
