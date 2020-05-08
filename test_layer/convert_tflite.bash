MODEL_DIR="/tf/minda/github/detect_ws/test_layer/out/mobile_block_test/save_models"
tflite_convert \
  --output_file=${MODEL_DIR}/test.tflite \
  --graph_def_file=${MODEL_DIR}/optimized_model.pb \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays='inputs' \
  --output_arrays='outputs' \
  --mean_values=114 \
  --std_dev_values=1 \
  --input_shapes=1,64,64,32 \
  --allow_nudging_weights_to_use_fast_gemm_kernel=true \
  --change_concat_input_ranges=false \
  --allow_custom_ops