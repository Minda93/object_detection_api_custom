import tensorflow as tf

from lib.layer.model_1 import model
from lib.layer.input_limit_model import input_limit_model

MODEL_MAP = {
  "model_1": model,
  "input_limit_model": input_limit_model,
}


class ModelBuilder(object):
  def __init__(self, model_name):
    self.model = MODEL_MAP[model_name]
  
  def build(self, inputs, image_size, num_classes, training=False):
    return self.model(inputs, image_size, num_classes, training)