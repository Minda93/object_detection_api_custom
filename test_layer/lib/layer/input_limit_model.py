import tensorflow as tf

from tensorflow.contrib import slim as contrib_slim
from object_detection.utils import ops
from object_detection.utils import shape_utils
slim = contrib_slim


def input_limit_model(inputs, input_size, output_size, training=False):
  
    # param
    normalizer_params = {
      'is_training': training,
      'decay': 0.9, 
      'center': True,
      'scale': True,
      'epsilon': 0.00001
    }
  
    with (slim.arg_scope([slim.conv2d, slim.separable_conv2d],\
                       activation_fn=tf.nn.relu6,\
                       normalizer_fn=slim.batch_norm,\
                       normalizer_params=normalizer_params)):
    
        #conv layer1
        net=slim.conv2d(inputs,output_size,[5,5],scope='conv1')

        return net
  

if __name__ == "__main__":
  pass