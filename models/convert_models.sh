# This script has to be run from the docker container started by ./docker_tflite2tensorflow.sh

FP=${1:-FP32}

source /opt/intel/openvino_2021/bin/setupvars.sh

# Palm detection model
tflite2tensorflow \
  --model_path palm_detection.tflite \
  --model_output_path palm_detection \
  --flatc_path ../../flatc \
  --schema_path ../../schema.fbs \
  --output_pb \
  --optimizing_for_openvino_and_myriad
# Generate Openvino "non normalized input" models: the normalization has to be mode explictly in the code
#tflite2tensorflow \
#  --model_path palm_detection.tflite \
#  --model_output_path palm_detection \
#  --flatc_path ../../flatc \
#  --schema_path ../../schema.fbs \
#  --output_openvino_and_myriad
# Generate Openvino "normalized input" models
/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo_tf.py --input_model palm_detection/model_float32.pb --model_name palm_detection_${FP} --data_type ${FP} --mean_values "[127.5, 127.5, 127.5]" --scale_values "[127.5, 127.5, 127.5]" --reverse_input_channels

# Hand landmark model
tflite2tensorflow \
  --model_path hand_landmark.tflite \
  --model_output_path hand_landmark \
  --flatc_path ../../flatc \
  --schema_path ../../schema.fbs \
  --output_pb
# Generate Openvino "non normalized input" models: the normalization has to be mode explictly in the code
# tflite2tensorflow \
#   --model_path hand_landmark.tflite \
#   --model_output_path hand_landmark \
#   --flatc_path ../../flatc \
#   --schema_path ../../schema.fbs \
#   --output_openvino_and_myriad
# Generate Openvino "normalized input" models
/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo_tf.py --input_model hand_landmark/model_float32.pb --model_name hand_landmark_${FP} --data_type ${FP} --scale_values "[255.0, 255.0, 255.0]" --reverse_input_channels
