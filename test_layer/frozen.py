import os
import sys
import shutil
import json
import argparse
import numpy as np

# tensorflow lib
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.python import ops
from tensorflow.tools.graph_transforms import TransformGraph

# lib 
from lib.utility import load_config
from lib.builder.model_builder import ModelBuilder

def parse_args():
  parser = argparse.ArgumentParser(description='detect process')
  parser.add_argument("--config_path",\
                      type = str,\
                      default="/tf/minda/github/detect_ws/cfg/test_layer/frozen.json",\
                      help="config path")
    
  parser.add_argument("--load_model",\
                      type = str,\
                      default= "",\
                      help = "load model.ckpt")
  
  parser.add_argument("--quantize",\
                      type = bool,\
                      default= True,\
                      help = "Weather use quantization aware training")
  
  args = parser.parse_args()

  return args

def mkdir(*directories):
  for directory in list(directories):
    if not os.path.exists(directory):
      os.makedirs(directory)
    else:
      pass

def freeze_model(saved_model_dir, output_node_names, output_filename):
  output_graph_filename = os.path.join(saved_model_dir, output_filename)
  initializer_nodes = ''
  freeze_graph.freeze_graph(
      input_saved_model_dir=saved_model_dir,
      output_graph=output_graph_filename,
      saved_model_tags = tag_constants.SERVING,
      output_node_names=output_node_names,
      initializer_nodes=initializer_nodes,
      input_graph=None,
      input_saver=False,
      input_binary=False,
      input_checkpoint=None,
      restore_op_name=None,
      filename_tensor_name=None,
      clear_devices=True,
      input_meta_graph=False,
  )

def frozen_graph(args, cfg, save_path, graph_path):
    
  with tf.Session(graph=tf.Graph()) as sess:
    # ========================
    # have bug
    
    # define input    
    x_image = tf.placeholder("float", shape=[None, *cfg["INPUT_SIZE"]], name=cfg["INPUT_NODE_NAMES"][0])
    
    model = ModelBuilder(cfg["MODEL_NAME"])
    # outputs = model.build(x_image, *cfg["INPUT_SIZE"] , *cfg["OUTPUT_SIZE"], training=False)
    outputs = model.build(x_image, cfg["INPUT_SIZE"] , cfg["OUTPUT_SIZE"][0], training=False)
    
    
    if(len(cfg["OUTPUT_NODE_NAMES"]) == 1):
      outputs = tf.identity(outputs, cfg["OUTPUT_NODE_NAMES"][0])
    
    # ========================
    
    if(args.quantize == True):
      g = tf.get_default_graph()
      tf.contrib.quantize.create_eval_graph(input_graph=g)
    
    saver = tf.train.Saver()
    
    if(args.load_model):
      saver.restore(sess, args.load_model)
    else:
      init = tf.global_variables_initializer()
      sess.run(init)
    
    # backup model
    saver.save(sess, save_path+"/model.ckpt")
    
    output_graph = save_path+"/{}".format(graph_path)
    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, # The session is used to retrieve the weights
      tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
      cfg["OUTPUT_NODE_NAMES"] # The output node names are used to select the usefull nodes
    ) 
    
    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
      f.write(output_graph_def.SerializeToString())
      print("%d ops in the final graph." % len(output_graph_def.node))
    
    return output_graph_def

def show_model(path):
  tf.train.import_meta_graph(path)
  for node in tf.get_default_graph().as_graph_def().node:
    print(node.name)

def get_graph_def_from_file(graph_filepath):
  tf.reset_default_graph()
  with ops.Graph().as_default():
    with tf.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def

def optimize_graph(model_dir, graph_filename, transforms, input_names, output_names, outname='optimized_model.pb'):
#   input_names = [input_name] # change this as per how you have saved the model
  graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))
  optimized_graph_def = TransformGraph(
      graph_def,
      input_names,  
      output_names,
      transforms)
  tf.train.write_graph(optimized_graph_def,
                      logdir=model_dir,
                      as_text=False,
                      name=outname)
  print('Graph optimized!')

def convert_graph_def_to_saved_model(export_dir, graph_filepath, input_name, outputs):
  graph_def = get_graph_def_from_file(graph_filepath)
  with tf.Session(graph=tf.Graph()) as session:
    tf.import_graph_def(graph_def, name='')
    tf.compat.v1.saved_model.simple_save(
        session,
        export_dir,# change input_image to node.name if you know the name
        inputs={input_name: session.graph.get_tensor_by_name('{}:0'.format(node.name))
            for node in graph_def.node if node.op=='Placeholder'},
        outputs={t.rstrip(":0"):session.graph.get_tensor_by_name(t) for t in outputs}
    )
    print('Optimized graph converted to SavedModel!')
    
def main():
  args = parse_args()
  cfg = load_config.readCfg(args.config_path)
  
  save_path = cfg["SAVE_MODEL_PATH"]+'/{}/save_models'.format(cfg["MODEL_NAME"])
  graph_path = "frozen_inference_graph.pb"
  mkdir(save_path)

  # freeze model
  frozen_graph(args, cfg, save_path, graph_path)
    
  
  # show model
  show_model(save_path+"/"+"model.ckpt.meta")
#   sys.exit()
    
  # Optimizing the graph via TensorFlow library
  transforms = []
  optimize_graph(save_path,
                 graph_path,
                 transforms,
                 cfg["INPUT_NODE_NAMES"],
                 cfg["OUTPUT_NODE_NAMES"], outname='optimized_model.pb')
  
  # convert this to a s TF Serving compatible mode
  outputs = []
  for node in cfg["OUTPUT_NODE_NAMES"]:
    outputs.append(node+":0")
  shutil.rmtree(save_path+'/save_model', ignore_errors=True)
  convert_graph_def_to_saved_model(save_path+'/save_model',
                                   save_path+'/optimized_model.pb', *cfg["INPUT_NODE_NAMES"], outputs)

if __name__ == "__main__":
  main()