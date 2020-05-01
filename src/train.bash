# ssd mobilenet v2 fpn r2_anchor_3 bdd
# CONFIG_FILE="/workspace/minda/github/detect_ws/cfg/train/ssdlite_mobilenet_v2_fpn_512_r2_anchor_3_bdd.config"
CONFIG_FILE="/workspace/minda/github/detect_ws/cfg/train/ssdlite_mobilenet_v2_fpn_512_r2_anchor_3_bdd_better.config"
WEIGHT_FILE="/workspace/minda/github/detect_ws/save_models/pytorch/ssdlite_mobilenet_v2_fpn6_r2_anchor_3_bdd/weight_tf_better.pickle"
LAYERS_FILE="/workspace/minda/github/detect_ws/save_models/pytorch/ssdlite_mobilenet_v2_fpn6_r2_anchor_3_bdd/layer_name_custom_better.txt"

MODEL_DIR="/workspace/minda/github/detect_ws/out/ssdlite_mobilenet_v2_fpn_512_r2_anchor_3_bdd"

CUDA_VISIBLE_DEVICES=0 python3 /workspace/minda/github/detect_ws/models/research/object_detection/legacy/train.py \
--logtostderr \
--pipeline_config_path=${CONFIG_FILE} \
--pytorch_weight_path=${WEIGHT_FILE} \
--pytorch_layers_path=${LAYERS_FILE} \
--load_pytorch=False \
--train_dir=${MODEL_DIR}