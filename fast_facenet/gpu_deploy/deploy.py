
import tvm

# tvm module for compiled functions.
loaded_lib = tvm.module.load('/root/net1.tar')
# json graph
loaded_json = open("/root/net1").read()
# parameters in binary
loaded_params = bytearray(open("/root/net1.params", "rb").read())

fcreate = tvm.get_global_func("tvm.graph_runtime.create")
ctx = tvm.gpu(0)
gmodule = fcreate(loaded_json, loaded_lib, ctx.device_type, ctx.device_id)
set_input, get_output, run = gmodule["set_input"], gmodule["get_output"], gmodule["run"]

gmodule["load_params"](loaded_params)

#set_input("x", tvm.nd.array(x_np))
#run()
#out = tvm.nd.empty(shape)
#get_output(0, out)
#print(out.asnumpy())
