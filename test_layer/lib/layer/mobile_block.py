import sys

import tensorflow as tf

from tensorflow.contrib import slim as contrib_slim
from object_detection.utils import ops
from object_detection.utils import shape_utils
slim = contrib_slim

def fixed_padding(inputs, kernel_size, rate=1):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    rate: An integer, rate for atrous convolution.
  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
  pad_total = kernel_size_effective - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                  [pad_beg, pad_end], [0, 0]])
  return padded_inputs

def mobile_block_v2(inputs, output_size, stride, expand_ratio, scope=None):
    
    with tf.variable_scope(scope, reuse=False) as scope:
        
        input_size = inputs.get_shape().as_list()[-1]
        padding = 'VALID'
    
        input_tensor = tf.identity(inputs, 'input')
        net = input_tensor
        
        if (expand_ratio != 1):
            net = slim.conv2d(
                  net,
                  int(round(input_size*expand_ratio)),
                  [1, 1],
                  padding=padding,
                  stride=1,
                  scope="pw")

        net = fixed_padding(net, kernel_size = 3, rate = stride)
        net = slim.separable_conv2d(
            net,
            None, [3, 3],
            depth_multiplier=1,
            padding=padding,
            stride=stride,
            scope="dw")
         
        net = slim.conv2d(
            net,
            output_size,
            [1, 1],
            padding=padding,
            stride=1,
            activation_fn=None,
            scope="pw_linear")
         
        if((stride == 1) and (input_size == output_size)):
            net += input_tensor
             
        return net
                
def mobile_block_test(inputs, input_size, output_size, training=False):
  
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
            
        
        net = mobile_block_v2(inputs, 32, 1, 6, "block_1")
        
        net = slim.conv2d(
            net,
            256,
            [1, 1],
            padding="VALID",
            stride=1,
            scope="reduce")
        
        net = slim.conv2d(
            net,
            63,
            [1, 1],
            padding="VALID",
            stride=1,
            scope="label")
        
        net = tf.reshape(net, (1,12288,21))
                
        return net
  

if __name__ == "__main__":
  pass