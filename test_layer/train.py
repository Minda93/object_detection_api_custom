import argparse
import numpy as np
import tensorflow as tf

# lib 
from lib.utility import load_config
from lib.builder.model_builder import ModelBuilder


def parse_args():
  parser = argparse.ArgumentParser(description='detect process')
  parser.add_argument("--config_path",\
                      type = str,\
                      default="/tf/minda/github/detect_ws/cfg/test_layer/test_layer.json",\
                      help="config path")

  parser.add_argument("--qunatize",\
                      type = bool,\
                      default= True,\
                      help = "Weather use qunantization aware training")

def mkdir(*directories):
  for directory in list(directories):
    if not os.path.exists(directory):
      os.makedirs(directory)
    else:
      pass

def run_train(args, cfg):
  with tf.Graph().as_default() as graph:
    pass


def main():
  args = parse_args()
  cfg = load_config.readCfg(args.config_path)

if __name__ == "__main__":
  main()