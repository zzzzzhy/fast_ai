from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import graph_util
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework.graph_util_impl import remove_training_nodes
from tensorflow.python.tools import optimize_for_inference_lib

import tensorflow as tf
import argparse
import os
import sys
import re
from six.moves import xrange  # @UnresolvedImport

def optimize_for_inference(input_graph_def, input_node_names, output_node_names,
                           placeholder_type_enum, toco_compatible=False):
  """Applies a series of inference optimizations on the input graph.
  Args:
    input_graph_def: A GraphDef containing a training model.
    input_node_names: A list of names of the nodes that are fed inputs during
      inference.
    output_node_names: A list of names of the nodes that produce the final
      results.
    placeholder_type_enum: The AttrValue enum for the placeholder data type, or
        a list that specifies one value per input node name.
    toco_compatible: Boolean, if True, only runs optimizations that result in
      TOCO compatible graph operations (default=False).
  Returns:
    An optimized version of the input graph.
  """
  #ensure_graph_is_valid(input_graph_def)
  optimized_graph_def = input_graph_def
  optimized_graph_def = optimize_for_inference_lib.strip_unused_lib.strip_unused(
      optimized_graph_def, input_node_names, output_node_names,
      placeholder_type_enum)
  optimized_graph_def = graph_util.remove_training_nodes(
      optimized_graph_def, output_node_names)
  optimized_graph_def = optimize_for_inference_lib.fold_batch_norms(optimized_graph_def)
  if not toco_compatible:
    optimized_graph_def = optimize_for_inference_lib.fuse_resize_and_conv(optimized_graph_def,
                                               output_node_names)
  #ensure_graph_is_valid(optimized_graph_def)
  return optimized_graph_def
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file
def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model metagraph and checkpoint
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = get_model_filenames(os.path.expanduser(args.model_dir))

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            model_dir_exp = os.path.expanduser(args.model_dir)
            saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file), clear_devices=True)
            tf.get_default_session().run(tf.global_variables_initializer())
            tf.get_default_session().run(tf.local_variables_initializer())
            saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))

            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()

            # Freeze the graph def

            #output_graph_def = freeze_graph_def(sess, input_graph_def, 'embeddings')
            #transformed_graph_def = remove_training_nodes(input_graph_def,output_node_names.split(","))


            transformed_graph_def = freeze_graph_def(sess, input_graph_def, 'embeddings')

            input_names = ["input","phase_train"]
            output_names = ["embeddings"]
            placeholder_type_enum = [tf.float32.as_datatype_enum,tf.bool.as_datatype_enum]
            transforms = [
                "remove_nodes(op=Identity, op=CheckNumerics)",
                #'strip_unused_nodes(type=float, shape="1,160,160,3", type=bool,shape="1")',
                #'fold_constants(ignore_errors=true)',
                #'fold_batch_norms',
                #'fold_old_batch_norms',
                'remove_control_dependencies'
            ]
            transformed_graph_def = TransformGraph(transformed_graph_def, input_names,
                                                   output_names, transforms)
            #transformed_graph_def = graph_util.remove_training_nodes(
            #    transformed_graph_def, output_names)

            transformed_graph_def = optimize_for_inference(
                transformed_graph_def,
                input_names,  # an array of the input node(s)
                output_names, # an array of the output nodes
                placeholder_type_enum,
                False)

            #transformed_graph_def = optimize_for_inference_lib.strip_unused_lib.strip_unused(
            #    transformed_graph_def, input_names, output_names,
            #    placeholder_type_enum)

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(args.output_file, 'wb') as f:
            f.write(transformed_graph_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(transformed_graph_def.node), args.output_file))

def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    # Get the list of important nodes
    whitelist_names = []
    blacklist_names = []

    previous = 'no'
    for node in input_graph_def.node:
        #if (node.name.startswith('InceptionResnet') or node.name.startswith('embeddings') or
        #        node.name.startswith('image_batch') or node.name.startswith('label_batch') or
        #        node.name.startswith('phase_train') or node.name.startswith('Logits')):
        if (node.name.startswith('InceptionResnet') or node.name.startswith('embeddings') or
                node.name.startswith('image_batch') or node.name.startswith('label_batch') or
                node.name.startswith('Logits')):
            try:
                firstLevel = node.name.split("/")[1]
                if firstLevel.startswith(previous) is not True:
                    print(node.name)
                    previous = firstLevel
            except:
                print(node.name)

            #if re.search('phase_train', node.name):
            #    print(node.name)
            whitelist_names.append(node.name)

    #for node in input_graph_def.node:
    #    print(node.name)
    #    blacklist_names.append(node.name)
    # Replace all the variables in the graph with constants of the same values
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_whitelist=whitelist_names)
    return output_graph_def

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model_dir', type=str,
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('output_file', type=str,
        help='Filename for the exported graphdef protobuf (.pb)')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
