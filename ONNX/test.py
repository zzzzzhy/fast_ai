import tensorflow as tf
from onnx_tf.frontend import tensorflow_graph_to_onnx_model

with tf.gfile.GFile("freezed.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    onnx_model = tensorflow_graph_to_onnx_model(graph_def,"embeddings",opset=0,
                                     producer_name="onnx-tensorflow",
                                     graph_name="graph",
                                     ignore_unimplemented=True)

    file = open("model.onnx", "wb")
    file.write(onnx_model.SerializeToString())
    file.close()
