import os
import sys
import tensorflow as tf

# lib 
from lib.builder.model_builder import ModelBuilder


def mkdir(*directories):
    for directory in list(directories):
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            pass

def show_model(path):
    tf.train.import_meta_graph(path)
    for node in tf.get_default_graph().as_graph_def().node:
        print(node.name)
                
def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    mkdir(absolute_model_dir+"/save_models")
    output_graph = absolute_model_dir + "/save_models/frozen_inference_graph.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
#         saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
        

        # define input
        x_image = tf.placeholder("float", shape=[None, 28 * 28])
        y_label = tf.placeholder("float", shape=[None, 10])
  
        # build model
        model = ModelBuilder("model_1")
        logits = model.build(x_image, 28 , 10, training=False)
        
        g = tf.get_default_graph()
        tf.contrib.quantize.create_eval_graph(input_graph=g)

        # We restore the weights
        saver = tf.train.Saver()
        saver.restore(sess, input_checkpoint)
        
        # backup ckpt
        save_path = saver.save(sess, os.path.dirname(os.path.realpath(output_graph))+"/model.ckpt")

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

   
def main():
    model_dir = "./out"
    
#     show_model(model_dir+"/"+"model.ckpt.meta")

    
    freeze_graph(model_dir, "Softmax")

if __name__ == "__main__":
    main()