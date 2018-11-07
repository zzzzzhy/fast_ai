import cv2
import tvm
import numpy as np
import time

from tvm.contrib import rpc, util, graph_runtime

# tvm module for compiled functions.
loaded_lib = tvm.module.load('./export/net2_android_mali.tar')
# json graph
loaded_json = open("./export/net2_android_mali").read()
# parameters in binary
loaded_params = bytearray(open("./export/net2_android_mali.params", "rb").read())

ctx = tvm.cl(0)

mod = graph_runtime.create(loaded_json, loaded_lib, ctx)
mod.load_params(loaded_params)

a = np.random.uniform(size=(1,3,112,112)).astype('float32')

print("first run, need calc graph")

start = time.time()
mod.run(data=a)
done = time.time()

out = mod.get_output(0)
print(out)

print('cost {}'.format(done-start))

start = time.time()
for i in range (1,100):
  step_start = time.time()
  mod.run(data=a)
  step_end = time.time()
  print('step {} cost {}'.format(i,(step_end - step_start)))
done = time.time()
print('everage {}'.format((done - start)/1000))

out = mod.get_output(0)
print(out)

#for i in range(0,100):


#set_input, get_output, run = gmodule["set_input"], gmodule["get_output"], gmodule["run"]
#gmodule["load_params"](loaded_params)

#set_input("x", tvm.nd.array(x_np))
#run()
exit(0)

ctx = tvm.gpu(0)
gmodule = fcreate(loaded_json, loaded_lib, ctx.device_type, ctx.device_id)
#set_input("x", tvm.nd.array(x_np))
#run()
#out = tvm.nd.empty(shape)
#get_output(0, out)
#print(out.asnumpy())
