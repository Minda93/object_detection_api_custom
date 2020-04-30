import tensorflow as tf

from tensorflow.contrib import slim as contrib_slim
from object_detection.utils import ops
from object_detection.utils import shape_utils
slim = contrib_slim


def model(inputs, image_size, num_classes, training=False):
  
    # param
    normalizer_params = {
    'is_training': training,
    'decay': 0.9, 
    'center': True,
    'scale': True,
    "epsilon": 0.00001
}
  
#   with (slim.arg_scope([slim.conv2d, slim.separable_conv2d],\
#                        activation_fn=tf.nn.relu6,\
#                        normalizer_fn=slim.batch_norm,\
#                        normalizer_params=normalizer_params)):
    
    x_image=tf.reshape(inputs,[-1,image_size,image_size,1])#shape of x is [N,28,28,1]
 
    #conv layer1
    net=slim.conv2d(x_image,32,[5,5],scope='conv1')#shape of net is [N,28,28,32]
    net=slim.max_pool2d(net,[2,2],scope='pool1')#shape of net is [N,14,14,32]
 
    #conv layer2
    net=slim.conv2d(net,64,[5,5],scope='conv2')#shape of net is [N,14,14,64]
    net=slim.max_pool2d(net,[2,2],scope='pool2')#shape of net is [N,7,7,64]
 
    #reshape for full connection
    net=tf.reshape(net,[-1,7*7*64])#[N,7*7*64]
 
    #fc1
    net = slim.fully_connected(net,1024,scope='fc1')#shape of net is [N,1024]
    
    #fc2
    net=slim.fully_connected(net,num_classes,scope='fc2')#[N,10]
    
    #softmax
    y=tf.nn.softmax(net)#[N,10]

    return y
  

if __name__ == "__main__":
  pass