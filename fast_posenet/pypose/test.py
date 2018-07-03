from pose import Pose as m
d = m()
d.init()

print(">>> test: ")
ret = d.detect()

print(">>> test again: ")
ret = d.detect()
