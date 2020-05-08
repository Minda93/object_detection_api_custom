# mobilenet_fpn6_mixnet_512_bdd_anchor_3
python3 /tf/minda/github/detect_ws/scripts/weight_transform.py \
--config-file '/tf/minda/github/detect_ws/save_models/pytorch/ssdlite_mobilenet_v2_fpn6_mixconv_anchor_3_bdd/mobilenet_fpn6_mixconv_512_bdd100k+TKU+CREDA+MOT17+taipei_E200-Copy1.yaml' \
--ckpt '/tf/minda/github/detect_ws/save_models/pytorch/ssdlite_mobilenet_v2_fpn6_mixconv_anchor_3_bdd/model_epoch190_better.pth' \
--layer_name_torch '/tf/minda/github/detect_ws/save_models/pytorch/ssdlite_mobilenet_v2_fpn6_mixconv_anchor_3_bdd/layer_name_custom.txt' \
--layer_name_tf '/tf/minda/github/detect_ws/save_models/pytorch/ssdlite_mobilenet_v2_fpn6_mixconv_anchor_3_bdd/layer_name_tf.txt' \
--save-path="/tf/minda/github/detect_ws/save_models/pytorch/ssdlite_mobilenet_v2_fpn6_mixconv_anchor_3_bdd/weight_tf_better.pickle"