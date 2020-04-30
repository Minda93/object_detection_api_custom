import sys
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

def representative_dataset_gen():
  for data in raw_test_data.take(100):
    image = data['image'].numpy()
    image = tf.image.resize(image, (320, 320))
    image = image[np.newaxis,:,:,:]
    yield [image]

def main():
    
  model_dir = './out/save_models/save_model'
  out_model_dir = './out/save_models'
  out_tflite_file = '/weight_quant.tflite'
  
  tf.compat.v1.enable_eager_execution()
  
  # Weight Quantization - Input/Output=float32
  converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
  converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
  tflite_quant_model = converter.convert()
  with open(out_model_dir+out_tflite_file, 'wb') as w:
    w.write(tflite_quant_model)
  print("Weight Quantization complete! - {}".format(out_tflite_file))
    
  # for post-training (quantization)


if __name__ == '__main__':
  main()