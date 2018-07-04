# some standard imports
import os
import nnvm
import tvm
import numpy as np

import tensorflow as tf
import nnvm.testing.tf

def get_feature(self, aligned):
  #face_img is bgr image
  #print(nimg.shape)
  input_blob = np.expand_dims(aligned, axis=0)
  data = mx.nd.array(input_blob)
  db = mx.io.DataBatch(data=(data,))

def convert(net, params, data_shape, dtype, target_host):
    # compile
    opt_level = 2 if dtype == 'float32' else 1
    with nnvm.compiler.build_config(opt_level=opt_level):
        print("Compiling")
        graph, lib, params = nnvm.compiler.build(
            net, tvm.target.mali(), shape={"image": data_shape}, params=params,
            dtype=dtype, target_host=target_host)
        print("Compiling Done")

    lib.export_library('./net1.tar')

    with open("./net1", "w") as fo:
        fo.write(graph.json())
    with open("./net1.params", "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(params))
model_name = 'frozen_model.pb'
with tf.gfile.FastGFile(os.path.join(
        "./", model_name), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        #graph_def = tf.GraphDef()
        graph = tf.import_graph_def(graph_def, name='')
        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
    # Call the utility to import the graph definition into default graph.
    graph_def = nnvm.testing.tf.ProcessGraphDefParam(graph_def)

    tf.import_graph_def(graph_def, name='')
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
print("Loading")
nnvm_net, nnvm_params = nnvm.frontend.from_tensorflow(graph_def)
print("Loading Done")
convert(nnvm_net, nnvm_params, (1,513, 513, 3) ,'float32','llvm --system-lib -target=aarch64-linux-gnu -mattr=+neon')
