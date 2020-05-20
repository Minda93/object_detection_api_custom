# object_detection_api_custom


# pytorch model weight to tensorflow model


## Environment

| tool                | version                   |
|---------------------|---------------------------|
| tensorflow-gpu      | 1.14 or 1.15              |
| tensorflow-datasets | 3.1.0 (tensorflow >= 1.15)|
| cuda                | 10.0.130                  |
| cudnn               | 7.4.1 (1.14) or 7.6 (1.15)|
| python              | 3.6.5                     |
| openCV              | 3.4.5.20                  |
| openCV-contrib      | 3.4.5.20                  |
| pip                 | 19.3.1 above              |
| protobuf-compiler   | 3.0.0                     |
| python-pil          | 7.1.1                     |
| python-lxml         | 4.5.0                     |
| tqdm                | 4.45.0                    |
| edgetpu_compiler    | 2.1.302470888             |
| edgetpu_runtime     | 13                        |
| tflite_runtime      |                           |

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
  $ protoc object_detection/protos/*.proto --python_out=.
```

3. 設定 PYTHONPATH 之環境變數
```bash
  $ vim ~/.bashrc

  * 在最後ㄧ行加入以下指令

  # for object deetection api 
  export TF_OBJECT_DETECTION="<workspace_path>/models/research:<workspace_path>/models/research/slim"
  export PYTHONPATH="${PYTHONPATH:-${TF_OBJECT_DETECTION}}"

  # 存檔後 初始環境
  $ source ~/.bashrc
```

4. 測試環境是否架好
```bash
  # 如果環境架設沒錯 會顯示 "OK"
  $ python object_detection/builders/model_builder_test.py
```
 
## Dataset preparing 
 
1. Label xml->csv using     
    1.1. 編輯 xml2csv_config.json
    * 強調資料集裡的 xml 須為 VOC 格式
    ```bash
      * label_path: 資料集的標註檔案位置(.xml)
      * out_path: 輸出csv檔案的位置與檔名(.csv)
    ``` 
    
    1.2. 執行
    ```bash
      $ python xml_to_csv.py --config_path <your_xml2csv_config_json_path>
    ```
2. csv to tfrecord 
    2.1. 編輯 generate_tfrecord_config.json
    ```bash
      * csv_path: 上個步驟輸出的csv檔案位置
      * img_path: 資料集的圖片路徑
      * out_path: 輸出record檔案的位置與檔名(.record)
    ```
    
    2.2. 編輯 generate_tfrecord.py 中的 categoryText2Int function
    * 需符合dataset class 格式
    ```py

      if label == "bike":
        return 1
      elif label == "bus":
        return 2
         .
         .
         .
    ```
    
    2.3. 執行
    ```bash
        $ python generate_tfrecord.py --config_path <your_generate_tfrecord_config_json_path>
    ```
3. Build your pbtxt, follow the style as below
    * 需符合dataset class 格式
    ```txt
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
```config
  train_config {
    # 訓練步數
    num_steps: 1

    # 批次大小(依照 GPU 記憶體大小調整)
    batch_size: 32 

    # 每步之批次大小 分批計算 loss 後整合更新權重
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

      # load multiple record file
      #input_path: ["train_a.record","train_b.record"]
    }
  }

  # 量化訓練
  graph_rewriter {
    quantization {
      # 量化統計 根據需求調整 通常等 float模型 穩定再執行
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
  
    2.1. 編輯 train.bash
    ```bash
      python <path_of_train.py>

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

1. 固化模型 
    1.1. 編輯 frozen_graph.bash
    ```bash
      python <path_of_export_inference_graph.py>
        
      # 訓練模型的參數檔位置
      --pipeline_config_path=<config_path>
        
      # 訓練模型的權重位置
      --trained_checkpoint_prefix=<ckpt_path>
        
      # 固化模型輸出位置
      --output_directory=<output_dir>
    ```
2. 測試模型 
    2.1. 編輯 detect_process.json
    ```json
      * PATH_FROZEN_GRAPH : 固化模型位置
      * PATH_TO_LABELS : dataset class label 位置
      * DATASET_NAME: our, voc case
      * NUM_CLASSES: class num
      * THRESHOLD_BBOX: bounding box 閥值

      * VIDEO_FILE : 測試影片位置
      * IMAGE_DATASET : 測試多張圖片位置
      * SINGE_IMAGE : 測試單張圖片位置

      * RESULT_OUT : 結果儲存位置

      # test mAP
      * USE_07_METRIC : 是否使用 voc 2007 evaluation
      * VAL_THRESHOLD : 驗證 bbox 的門檻
      * VAL_MAP :  用於驗證精準度之測試集位置
      * VAL_MAP_OUT : 輸出驗證結果

      # 尚未驗證
      * PATH_TFLITE : tflite 模型位置
      * PATH_TPU : edgetpu 模型位置
    ```
   2.2. 執行 demo
   ```bash
     $ python detect_process.py
        
     # args
       # 參數檔位置
       --config_path = <path_of_detect_process.json>

       # 選擇模型 <graph tflite tpu>
       --engine=graph

       # 測試模式 <video, image, images>
       --mode=video

       # 儲存圖片
       --save=false

       # 顯示圖片
       --show=false
    ```
    
## Evaluation model
1. 生成驗證資料 
    1.1. 編輯 parse_voc_xml.json
    ```json
      * PATH_TO_DATASET : 驗證資料集位置 (voc 格式)
      * PATH_TO_LABELS : dataset class label 位置
      * OUT_PATH : 輸出驗證資料位置
    ```
    1.2. 生成
    ```bash
      $ python parse_voc_xml.py
        
      # args
        # 參數檔位置
        --config_path = <path_of_parse_voc_xml.json>
    ```
2. 執行驗證 
    2.1. 編輯 detect_process.json
    ```json
      * PATH_FROZEN_GRAPH : 固化模型位置
      * PATH_TO_LABELS : dataset class label 位置
      * DATASET_NAME: our, voc case
      * NUM_CLASSES: class num
      * THRESHOLD_BBOX: bounding box 閥值
        
      * VIDEO_FILE : 測試影片位置
      * IMAGE_DATASET : 測試多張圖片位置
      * SINGE_IMAGE : 測試單張圖片位置
        
      * RESULT_OUT : 結果儲存位置
        
      # test mAP
      * USE_07_METRIC : 是否使用 voc 2007 evaluation
      * VAL_THRESHOLD : 驗證 bbox 的門檻
      * VAL_MAP :  用於驗證精準度之測試集位置
      * VAL_MAP_OUT : 輸出驗證結果
        
      # 尚未驗證
      * PATH_TFLITE : tflite 模型位置
      * PATH_TPU : edgetpu 模型位置
    ```
    2.2. 執行
    ```bash
      $ python detect_process.py
        
      # args
        # 參數檔位置
        --config_path = <path_of_detect_process.json>

        # 選擇模型 <graph tflite tpu>
        --engine=graph

        # 驗證模式 map
        --mode=map
    ```

## Mkdir folder
1. dataset
    * tfrecord : tensorflow model train use tfrecord
2. save_models
    * pytorch : pytorch model ( weight, ckpt, layer_name_custom.txt, layer_name_tf.txt)  
    * tensorflow : 通常會放讀完權重的模型 尚未訓練的 (權重讀取在docker中需要數十分鐘, 因此方便下次直接讀取模型)
3. out 


## Test model 
1. ssdlite_mobilenet_v2_fpn_512_r2_anchor_3_bdd_15.config
2. ssdlite_mobilenet_v2_fpn6_512_mixconv_anchor_3_bdd_better.config

## DOING
1. 架設專門測試 neural network block 等程式
2. tf15 量化訓練
    * tflite model 轉換 edgetpu model 準確率會下降 5% 

## TO DOO
1. 新增權重載點
2. 量化訓練調整 
    * 直接進行量化轉換在 tf15 環境上，效果比在 tf14 還要好 
    * tf15 量化訓練，還在測試中
3. 增加 config file 教學 
4. 測試引擎 (tflite, tpu)
    * tflite on tf15 is ok
5. 新增架設 tflite, edgetpu env 教學
6. 新增 rfb
    * rfb 無法量化
7. 待測試
    * 空洞卷積
    * batchnorm 與 relu 可加的位置
    * 支援1\*k k\*1 模型
    * resnet 限制

## Questions
1. load pytorch weight to tensorflow model: mAP 會下降 
    * 主要原因加入fake quantization node
2. fpn7 tflite model to tpu model 精準度大幅下降 
    * edgetpu compiler 可能要等官方更新
3. tflite model 轉換 edgetpu model 準確率會下降 5%
    * mixconv

## Reference
 * [object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection)
 * [lufficc pytorch ssd](https://github.com/lufficc/SSD?fbclid=IwAR2WFi1g6gbpH8GzSBBO-ERHTUIX7VXbPbTtK5Z-kIT1h-dSWlx3GEHkkqc)
 * [object detection api setting](https://blog.gtwang.org/programming/tensorflow-object-detection-api-tutorial/)
 * [google api guide](https://github.com/AcgEuSmile/Tensorboard_object_detection_api)
 * [YOLOv3_TensorFlow](https://github.com/wizyoung/YOLOv3_TensorFlow)
 * [freeze a model](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc) 
 * [PINTO for Full Integer Quantization](https://qiita.com/PINTO/items/1312d308b553362a8ebf)