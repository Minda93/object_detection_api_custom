# object_detection_api_custom


# ssd_mobilenet_v2_fpn_r2_anchor_3


## Environment

| tool              | version  |
|-------------------|----------|
| tensorflow-gpu    | 1.14     |
| cuda              | 10.0.130 |
| cudnn             | 7.4.1    |
| python            | 3.6.5    |
| openCV            | 3.4.5.20 |
| openCV-contrib    | 3.4.5.20 |
| pip               | 9.0.3    |
| protobuf-compiler | 3.0.0    |
| python-pil        | |
| python-lxml       | |

## Object detection api setting

1. 安裝相依套件
```bash
  $ sudo apt-get install protobuf-compiler python-pil python-lxml
  $ pip install pillow
  $ pip install lxml
```
2. env setting command:
```bash
  # 移動到你的工作空間底下
  $ cd <workspace_path>
     
  # 移動到該資料夾底下
  $ cd models/research/
  
  # 編譯protobuf
 protoc object_detection/protos/*.proto --python_out=.
```
3. 設定 PYTHONPATH 之環境變數
```bash
  $ vim ~/.bashrc
     
  * 在最下行加入以下幾行
     
  # for object deetection api 
  export TF_OBJECT_DETECTION="<workspace_path>/models/research:<workspace_path>/models/research/slim"
  export PYTHONPATH="${PYTHONPATH:-${TF_OBJECT_DETECTION}}"
```
 
## Dataset preparing
1. Label xml->csv using
  A. 編輯 xml2csv_config.json
    * 強調資料集裡的 xml 須為 VOC 格式
```bash
  * label_path: 資料集的標註檔案位置(.xml)
  * out_path: 輸出csv檔案的位置與檔名(.csv)
```
  B. 執行
```bash
  $ python scripts/xml_to_csv.py --config_path <your_xml2csv_config_json_path>
```
2. csv to tfrecord
  A. 編輯 generate_tfrecord_config.json
```bash
  * csv_path: 上個步驟輸出的csv檔案位置
  * img_path: 資料集的圖片路徑
  * out_path: 輸出record檔案的位置與檔名(.record)
```
  B. 編輯 generate_tfrecord.py 中的 categoryText2Int function
    * 需符合dataset class 格式
```bash
  if label == "bike":
    return 1
  elif label == "bus":
    return 2
     .
     .
     .
```
  C. 執行
```bash
 $ python scripts/generate_tfrecord.py --config_path <your_generate_tfrecord_config_json_path>
```
3. Build your pbtxt, follow the style as below
  * 需符合dataset class 格式
```bash
  item{
    id: 1
    name: 'bike'
  }
  item{
    id: 2
    name: 'bus'
  }
     .
     .
     .
```

## Load pytorch weight
1. 編輯 config file
```bash
  train_config {
    # 訓練步數
    num_steps: 1
   
    # 批次大小(依照 GPU 記憶體大小調整)
    batch_size: 32 
    
    # 每步之批次大小 分批計算 loss 後整合計算
    sync_replicas: true
    startup_delay_steps: 0.0
    replicas_to_aggregate: 8
    
    # load cpkt weight (also open load pytorch weight)
    fine_tune_checkpoint: "MODEL_CKPT_PATH"
    load_all_detection_checkpoint_vars: true
    fine_tune_checkpoint_type: "detection"
    from_detection_checkpoint: true
  } 
  
  train_input_reader {
    # dataset class label
    label_map_path: "label.pbtxt"
    
    # load dataset to train
    tf_record_input_reader {
      input_path: "trainval.record"
    }
  }
  
  # 量化訓練
  graph_rewriter {
    quantization {
      # 量化統計 根據需求調整 通常等 float模型 穩定在執行
      delay: 1000
      
      weight_bits: 8
      activation_bits: 8
    }
  }
```
2. 生成 tensorflow 模型
 * 強調 pytorch model 必須與 tensorflow model "一模一樣" 才能讀取 weight
 * pytorch model 請參考 lufficc pytorch ssd
 * 讀取權重是利用 "pickle 檔" 讀取
 
 A. 編輯 train.bash
```bash
    python3 <path_of_train.py>
    
    # 輸出模型位置
    --train_dir=<output_path>
    
    # 模型參數檔位置
    --pipeline_config_path=<config_path>
    
    # pytorch 權重位置
    --pytorch_weight_path=<pickle_path>
    
    # pytorch all layer names (按照 tensorflow model 順序)
    --pytorch_layers_path=<path_of_all_layer_name.txt>
    
    # 讀取 pytorch 權重 (fine_tune_checkpoint 需開啟)
    --load_pytorch=True
```
## Demo tensorflow model
1. 

## TO DOO
1. 新增權重載點
2. 調整 tensorflow model 計算 mAP 下降 2 % (主要原因加入fake quantion node)
3. 量化訓練調整
4. 增加 config file 教學

## Reference
 * [object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection)
 * [lufficc pytorch ssd](https://github.com/lufficc/SSD?fbclid=IwAR2WFi1g6gbpH8GzSBBO-ERHTUIX7VXbPbTtK5Z-kIT1h-dSWlx3GEHkkqc)
 * [object detection api setting](https://blog.gtwang.org/programming/tensorflow-object-detection-api-tutorial/)
 * [google api guide](https://github.com/AcgEuSmile/Tensorboard_object_detection_api)