# ssd mobilenet v2 fpn r2_anchor_3 bdd
CONFIG_FILE="/workspace/minda/github/detect_ws/cfg/train/ssdlite_mobilenet_v2_fpn_512_r2_anchor_3_bdd.config"
MODEL_DIR="/workspace/minda/github/detect_ws/out/ssdlite_mobilenet_v2_fpn_512_r2_anchor_3_bdd"
WEIGHT_FILE="/workspace/minda/github/detect_ws/save_models/pytorch/ssdlite_mobilenet_v2_fpn_r2_anchor_3_bdd/weight_tf.pickle"
LAYERS_FILE="/workspace/minda/github/detect_ws/save_models/pytorch/ssdlite_mobilenet_v2_fpn_r2_anchor_3_bdd/layer_name_custom.txt"

mkdir -p logs/
now=$(date +"%Y_%m_%d")
CUDA_VISIBLE_DEVICES=0 python3 /workspace/minda/github/detect_ws/models/research/object_detection/legacy/train.py \
--logtostderr \
--pipeline_config_path=${CONFIG_FILE} \
--pytorch_weight_path=${WEIGHT_FILE} \
--pytorch_layers_path=${LAYERS_FILE} \
--load_pytorch=False \
--train_dir=${MODEL_DIR} 2>&1 | tee logs/train_$now.txt