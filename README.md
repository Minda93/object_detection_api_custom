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
 * env setting
 ```bash
     $ vim ~/.bashrc
     
     * add OBJECT_DETECION_API_PATH to python path
     
     # for object deetection api 
     export TF_OBJECT_DETECTION="OBJECT_DETECION_API_PATH/research:OBJECT_DETECION_API_PATH/research/slim"
     export PYTHONPATH="${PYTHONPATH:-${TF_OBJECT_DETECTION}}"
 ```

## Reference
 * [object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection)
 * [lufficc pytorch ssd](https://github.com/lufficc/SSD?fbclid=IwAR2WFi1g6gbpH8GzSBBO-ERHTUIX7VXbPbTtK5Z-kIT1h-dSWlx3GEHkkqc)
 * [object detection api setting](https://blog.gtwang.org/programming/tensorflow-object-detection-api-tutorial/)
 * [ＧＯＯＧＬＥ　ＡＰＩ　ＧＵＩＤＥＳ](https://github.com/AcgEuSmile/Tensorboard_object_detection_api)