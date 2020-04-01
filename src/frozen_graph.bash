# ssd mobilenet v2 fpn r2_anchor_3 bdd
MODEL_DIR="/workspace/minda/github/detect_ws/out/ssdlite_mobilenet_v2_fpn_512_r2_anchor_3_bdd"
CKPT_STEP="0"

# creates the frozen inference graph in fine_tune_model for test
python3 /workspace/minda/github/detect_ws/models/research/object_detection/export_inference_graph.py \
--input_type=image_tensor \
--pipeline_config_path=${MODEL_DIR}/pipeline.config \
--trained_checkpoint_prefix=${MODEL_DIR}/model.ckpt-${CKPT_STEP} \
--output_directory=${MODEL_DIR}/save_model