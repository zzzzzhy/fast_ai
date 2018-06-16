
import nnvm
import tvm

from pickle import load

print('1')

with open('test.pkl', 'rb') as f:
    print('2')
    mx_sym = load(f)
    print('3')

print('4')
# now we use the same API to get NNVM compatible symbol
nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)
# repeat the same steps to run this model using TVM

print('5')
######################################################################
# now compile the graph
import nnvm.compiler
target = 'llvm -target=aarch64-linux-gnu -mattr=+neon'
shape_dict = {'data': (1, 3, 224, 224)}
graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now, we would like to reproduce the same forward computation using TVM.
from tvm.contrib import graph_runtime
ctx = tvm.gpu(0)
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input('data', tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
tvm_output = m.get_output(0, tvm.nd.empty((1000,), dtype))
top1 = np.argmax(tvm_output.asnumpy())
print('TVM prediction top-1:', top1, synset[top1])


