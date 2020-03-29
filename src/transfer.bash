# ssd mobilenet v2 fpn r2_anchor_3 bdd
# CONFIG_FILE="/workspace/minda/github/detect_ws/cfg/train/ssdlite_mobilenet_v2_fpn_512_r2_anchor_3_bdd_test.config"
# MODEL_DIR="/workspace/minda/github/detect_ws/out/ssdlite_mobilenet_v2_fpn_512_r2_anchor_3_bdd_test"
MODEL_DIR="/workspace/minda/github/detect_ws/out/ssdlite_mobilenet_v2_fpn_512_r2_anchor_3_bdd"
# MODEL_DIR="/workspace/minda/github/detect_ws/out/ssdlite_mobilenet_v2_fpn7_512_r2_anchor_3_bdd"
CKPT_STEP="0"


# create the tensorflow lite graph
python3 /workspace/minda/github/detect_ws/models/research/object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=${MODEL_DIR}/pipeline.config \
--trained_checkpoint_prefix=${MODEL_DIR}/model.ckpt-${CKPT_STEP} \
--output_directory=${MODEL_DIR}/tflite \
--add_postprocessing_op=true \
--max_detections=100


# CONVERTING frozen graph to quantized TF Lite file...
tflite_convert \
  --output_file=${MODEL_DIR}/tflite/mobilenet.tflite \
  --graph_def_file=${MODEL_DIR}/tflite/tflite_graph.pb \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays='normalized_input_image_tensor' \
  --output_arrays='TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3' \
  --mean_values=114 \
  --std_dev_values=1 \
  --input_shapes=1,512,512,3 \
  --allow_nudging_weights_to_use_fast_gemm_kernel=true \
  --change_concat_input_ranges=false \
  --allow_custom_ops
  # --default_ranges_min=0 \
  # --default_ranges_max=6

# tflite_convert \
#   --output_file=${MODEL_DIR}/tflite/mobilenet.tflite \
#   --graph_def_file=${MODEL_DIR}/tflite/tflite_graph.pb \
#   --inference_type=FLOAT \
#   --input_arrays='normalized_input_image_tensor' \
#   --output_arrays='TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3' \
#   --mean_values=127 \
#   --std_dev_values=127 \
#   --input_shapes=1,512,512,3 \
#   --allow_nudging_weights_to_use_fast_gemm_kernel=true \
#   --change_concat_input_ranges=false \
#   --allow_custom_ops