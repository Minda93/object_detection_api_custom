import tensorflow as tf
import numpy as np

# dataset
from tensorflow.examples.tutorials.mnist import input_data

# lib 
from lib.builder.model_builder import ModelBuilder

IMAGE_SIZE  = 28
NUM_CLASSES = 10

# graph train op
def Accuracy(logits, labels):
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.arg_max(labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  
  return accuracy

def Training(loss, learning_rate):
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
  
  return train_step

def Loss(logits, labels):
  cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
  
  return cross_entropy


# run train
def Run_Train(dataset):
  with tf.Graph().as_default() as graph:
    # define input
    x_image = tf.placeholder("float", shape=[None, IMAGE_SIZE * IMAGE_SIZE])
    y_label = tf.placeholder("float", shape=[None, NUM_CLASSES])
  
    # build model
    model = ModelBuilder("model_1")
    logits = model.build(x_image, IMAGE_SIZE , NUM_CLASSES, training=True)

    # define op
    loss_value = Loss(logits, y_label)
    
    tf.contrib.quantize.create_training_graph(input_graph=graph, quant_delay=0)
    
    train_op = Training(loss_value,1e-4) 
    accur = Accuracy(logits, y_label) 
    
    # saver 
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
  
    with tf.Session() as sess:
      sess.run(init)
      print("start train")
      for epoch in range(1000):
        batch = dataset.train.next_batch(50)
    
        if(epoch % 100 == 0):
          train_accury = sess.run(accur, feed_dict={x_image: batch[0], y_label: batch[1]})        
          print('step %d,training accuracy  %g !!!!!!!' % (epoch,train_accury))
      
        sess.run(train_op, feed_dict={x_image: batch[0], y_label: batch[1]})
      
        
      print("train finish")
    
      test_batch = dataset.test.next_batch(500, shuffle=False)
      print("test accuracy : %g" % sess.run(accur, feed_dict={x_image: test_batch[0], y_label: test_batch[1]}))
    
      save_path = saver.save(sess, "./out/model.ckpt")
      print('Save END : ' + save_path )
        
    
def main():
  print("Load mnist dataset")
  mnist = input_data.read_data_sets('../dataset/MNIST_DATA',one_hot=True)
  print("Load mnist finish")

  Run_Train(mnist)

if __name__ == "__main__":
  main()