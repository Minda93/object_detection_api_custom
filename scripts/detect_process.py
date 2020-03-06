import sys
import os
import json
import argparse
import glob
import numpy as np
import time
import cv2
from PIL import Image

""" lib """
from object_detection.utils import label_map_util
from lib.utility import load_config

DELAY_TIME = 2000

def parse_args():
  parser = argparse.ArgumentParser(description='detect process')
  parser.add_argument("--config_path",\
                      type = str,\
                      default="/workspace/minda/github/detect_ws/cfg/demo/detect_process.json",\
                      help="config path")

  parser.add_argument("--engine",\
                      type = str,\
                      default="graph",\
                      help = "choose model (graph, tflite, tpu)")
  
  parser.add_argument("--mode",\
                      type = str,\
                      default="video",\
                      help = "choose model (video, image, images, map)")
  
  parser.add_argument("--save",\
                      type = bool,\
                      default=False,\
                      help = "save result for mode (video, image, images)")
  
  parser.add_argument("--show",\
                      type = bool,\
                      default= True,\
                      help = "show for mode (video, image, images)")
  
  args = parser.parse_args()

  return args

def mkdir(*directories):
  for directory in list(directories):
    if not os.path.exists(directory):
      os.makedirs(directory)
    else:
      pass

""" color """
ColorTable = dict({'RED': (0, 0, 255),\
                  'ORANGE': (0, 165, 255),\
                  'YELLOW': (0, 255, 255),\
                  'GREEN': (0, 255, 0),\
                  'BLUE': (255, 127, 0),\
                  'INDIGO': (255, 0, 0),\
                  'PURPLE': (255, 0, 139),\
                  'WHITE': (255, 255, 255),\
                  'BLACK': (0, 0, 0)}
)
ClassColor = dict(
        default = {'bike': ColorTable['RED'],
                   'bus': ColorTable['ORANGE'],
                   'car': ColorTable['YELLOW'],
                   'motor': ColorTable['GREEN'],
                   'person': ColorTable['WHITE'],
                   'rider': ColorTable['INDIGO'],
                   'truck': ColorTable['PURPLE'],
                  }
)

def draw_BBox(frame,bboxes,min_score_thresh = 0.2,dataset="our"):
  if(dataset == "our"):
    for bbox in bboxes:
      if(bbox['score'] >= min_score_thresh):
        cv2.rectangle(frame,\
                      (bbox['bbox']['xmin'], bbox['bbox']['ymax']),\
                      (bbox['bbox']['xmax'], bbox['bbox']['ymin']),\
                      ClassColor['default'][bbox['id']], 2)
        
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale = 1
        thickness = 1
        margin = 5
        size = cv2.getTextSize(bbox['id'], font, font_scale, thickness)
        text_width = size[0][0]
        text_height = size[0][1]
        cv2.rectangle(frame, (bbox['bbox']['xmin'], bbox['bbox']['ymax']),
                      (bbox['bbox']['xmin']+text_width, bbox['bbox']['ymax']-text_height),
                      (0, 0, 0), thickness = -1)
        
        cv2.putText(frame, bbox['id'], (bbox['bbox']['xmin'], bbox['bbox']['ymax']),
                    font, 1, ClassColor['default'][bbox['id']], 1, cv2.LINE_AA)
  else:
    for bbox in bboxes:
      if(bbox['score'] >= min_score_thresh):
        print(bbox)
        cv2.rectangle(frame,\
                      (bbox['bbox']['xmin'], bbox['bbox']['ymax']),\
                      (bbox['bbox']['xmax'], bbox['bbox']['ymin']),\
                      ColorTable['GREEN'], 2)
        
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale = 1
        thickness = 1
        margin = 5
        size = cv2.getTextSize(bbox['id'], font, font_scale, thickness)
        text_width = size[0][0]
        text_height = size[0][1]
        cv2.rectangle(frame, (bbox['bbox']['xmin'], bbox['bbox']['ymax']),
                      (bbox['bbox']['xmin']+text_width, bbox['bbox']['ymax']-text_height),
                      (0, 0, 0), thickness = -1)
        
        cv2.putText(frame, bbox['id'], (bbox['bbox']['xmin'], bbox['bbox']['ymax']),
                    font, 1, ColorTable['GREEN'], 1, cv2.LINE_AA)

def test_mAP(cfg,engine,args):
  test_annos = dict()
  for index, image_path in enumerate(glob.glob(cfg['VAL_MAP']+'/*.jpg')):
    image_id = image_path.rstrip().split('/')[-1]

    image = Image.open(image_path)
    image_np = np.array(image).astype(np.uint8)
    im_height, im_width, _ = image_np.shape
    # image = cv2.imread(image_path,cv2.IMREAD_COLOR)
    # image_np = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # im_height, im_width, _ = image.shape

    if(args.engine == 'graph'):
      frame_expanded = np.expand_dims(image_np, axis=0)
    elif(args.engine == 'tflite'):
      frame_resize = cv2.resize(image_np,(512,512))
      frame_expanded = np.expand_dims(frame_resize, axis=0)
    elif(args.engine == 'tpu'):
      frame_expanded = image
    
    bboxes, _ = engine.Run(frame_expanded,im_width, im_height)
    test_annos[image_id] = {'objects':bboxes}
  
  test_annos = {'imgs': test_annos}
  fd = open(cfg['VAL_MAP_OUT'], 'w')
  json.dump(test_annos, fd)
  fd.close()
  print("success")

def show_image(cfg,engine,args,mode = 'single'):

  if(args.show):
    # cv2.namedWindow('FRAME',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('FRAME', 600,600)
    pass

  if(mode == 'single'):
    frame = cv2.imread(cfg['SINGE_IMAGE'],cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    im_height, im_width, _ = frame.shape

    if(args.engine == 'graph'):
      frame_expanded = np.expand_dims(frame_rgb, axis=0)
    elif(args.engine == 'tflite'):
      frame_resize = cv2.resize(frame_rgb,(512,512))
      # frame_resize = (frame_resize - np.array([123, 117, 104])).astype(np.uint8)
      # frame_resize = frame_resize.astype(np.float32)
      frame_expanded = np.expand_dims(frame_resize, axis=0)
    elif(args.engine == 'tpu'):
      frame_expanded = Image.fromarray(frame_rgb)

    bboxes, num_detections = engine.Run(frame_expanded, im_width, im_height)
    
    draw_BBox(frame,bboxes,cfg['THRESHOLD_BBOX'],dataset=cfg['DATASET_NAME'])

    if(args.show):
      cv2.imshow("FRAME", frame)
      cv2.waitKey(DELAY_TIME)

    if(args.save):
      image_file = cfg['SINGE_IMAGE'].split('/')[-1]
      save_path = cfg['RESULT_OUT']+'/image/{}/'.format(args.engine)
      mkdir(save_path)
      cv2.imwrite(save_path+image_file, frame)

  elif(mode == 'all'):
    for image_file in os.listdir(cfg['IMAGE_DATASET']):
      frame = cv2.imread(cfg['IMAGE_DATASET']+'/'+image_file,cv2.IMREAD_COLOR)
      frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      im_height, im_width, _ = frame.shape

      if(args.engine == 'graph'):
        frame_expanded = np.expand_dims(frame_rgb, axis=0)
      elif(args.engine == 'tflite'):
        frame_resize = cv2.resize(frame_rgb,(512,512))
        # frame_resize = frame_resize.astype(np.float32)
        frame_expanded = np.expand_dims(frame_resize, axis=0)
      elif(args.engine == 'tpu'):
        frame_expanded = Image.fromarray(frame_rgb)

      bboxes, num_detections = engine.Run(frame_expanded, im_width, im_height)

      draw_BBox(frame,bboxes,cfg['THRESHOLD_BBOX'],dataset=cfg['DATASET_NAME'])

      if(args.show):
        cv2.imshow("FRAME", frame)
        cv2.waitKey(DELAY_TIME)

      if(args.save):
        dataset = cfg['IMAGE_DATASET'].split('/')[-3]
        direction = cfg['IMAGE_DATASET'].split('/')[-1]
        save_path = cfg['RESULT_OUT']+'/image/{}/{}/{}/'.format(args.engine,dataset,direction)
        mkdir(save_path)
        cv2.imwrite(save_path+image_file, frame)

  cv2.destroyAllWindows()
  print('finish')

def show_video(cfg,engine,args):
  """ init """
  log_inference_time = []
  log_fps = []
  frame_counter = 0

  """ video """
  camera = cv2.VideoCapture(cfg['VIDEO_FILE'])
  if(args.save):
    video_file = cfg['VIDEO_FILE'].split('/')[-1]
    save_path = cfg['RESULT_OUT']+'/video/{}/'.format(args.engine)
    video_out = save_path + video_file

    mkdir(save_path)

    sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out,fourcc,30, sz, True)

  if(args.show):
    # cv2.namedWindow('FRAME',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('FRAME', 640,480)
    pass

  while(camera.isOpened()):
    (grabbed, frame) = camera.read()
    if(grabbed == True):
      frame_counter += 1

      start = time.time()
      im_height, im_width, _ = frame.shape
      frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      
      if(args.engine == 'graph'):
        frame_expanded = np.expand_dims(frame_rgb, axis=0)
      elif(args.engine == 'tflite'):
        frame_resize = cv2.resize(frame_rgb,(512,512))
        frame_expanded = np.expand_dims(frame_resize, axis=0)
      elif(args.engine == 'tpu'):
        frame_expanded = Image.fromarray(frame_rgb)

      s_inference = time.time()
      bboxes, num_detections = engine.Run(frame_expanded, im_width, im_height)

      end_inference = time.time()
      inference_time = end_inference - s_inference
      print('inference time {}'.format(inference_time))

      draw_BBox(frame,bboxes,cfg['THRESHOLD_BBOX'],dataset=cfg['DATASET_NAME'])

      end = time.time()
      seconds = end - start
      fps_rate = 1 / seconds
      cv2.putText( frame, "FPS:{}".format(round(fps_rate,1)),(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255, 0, 0), 2)

      if(frame_counter > 30):
        log_inference_time.append(inference_time)
        log_fps.append(fps_rate)

      if(args.show):
        cv2.imshow("FRAME", frame)
      
      if(args.save):
        out.write(frame)

    else:
      print("no video")
      break
      
    key = cv2.waitKey(1)
    if(key==113):
        sys.exit(0)
  
  print("inference time : ",sum(log_inference_time)/len(log_inference_time))
  print("fps : ",sum(log_fps)/len(log_fps))

  camera.release()
  cv2.destroyAllWindows()
  if(args.save):
    out.release()


def main():
  args = parse_args()
  cfg = load_config.readCfg(args.config_path)

  """ label """
  label_map = label_map_util.load_labelmap(cfg["PATH_TO_LABELS"])
  categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes=cfg['NUM_CLASSES'], use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  
  if(args.engine == 'graph'):
    from lib.engine.float_engine import Engine
    engine = Engine(cfg,category_index)
  elif(args.engine == 'tflite'):
    from lib.engine.tflite_engine import Engine
    engine = Engine(cfg,category_index)
  elif(args.engine == 'tpu'):
    from lib.engine.tpu_engine import Engine
    engine = Engine(cfg,category_index)
  
  if(args.mode == 'video'):
    show_video(cfg,engine,args)
  elif(args.mode == 'image'):
    show_image(cfg,engine,args,'single')
  elif(args.mode == 'images'):
    show_image(cfg,engine,args,'all')
  elif(args.mode == 'map'):
    test_mAP(cfg,engine,args)

if __name__ == "__main__":
  main()
