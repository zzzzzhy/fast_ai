import copy

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2

INPUT_GRAPH_DEF_FILE = './freezed.pb'
OUTPUT_GRAPH_DEF_FILE = './freezed_clean.pb'

# Get NodeDef of a constant tensor we want to put in place of
# the placeholder.
# (There is probably a better way to do this)
example_graph = tf.Graph()
with tf.Session(graph=example_graph):
    #c = tf.constant(False, dtype=bool, shape=[], name='phase_train')
    c = tf.constant(0.0, dtype=tf.float32, shape=[], name='phase_train')
    for node in example_graph.as_graph_def().node:
        if node.name == 'phase_train':
            c_def = node

# load our graph
#graph = tf.load_graph(INPUT_GRAPH_DEF_FILE)
with gfile.FastGFile(INPUT_GRAPH_DEF_FILE,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
#graph_def = graph.as_graph_def()

# Create new graph, and rebuild it from original one
# replacing phase train node def with constant
new_graph_def = graph_pb2.GraphDef()
for node in graph_def.node:
    if node.name == 'phase_train':
        new_graph_def.node.extend([c_def])
    else:
        new_graph_def.node.extend([copy.deepcopy(node)])

# save new graph
with tf.gfile.GFile(OUTPUT_GRAPH_DEF_FILE, "wb") as f:
    f.write(new_graph_def.SerializeToString())
