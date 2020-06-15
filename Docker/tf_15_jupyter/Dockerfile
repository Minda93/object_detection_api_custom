# ================================================
# 
# for x86 computer
# tensorflow 1.15 gpu jupyter
# 
# edgetpu python api 
# https://coral.ai/software/#debian-packages
#
# edgetpu github
# https://github.com/google-coral/edgetpu
# ================================================

FROM tensorflow/tensorflow:1.15.2-gpu-jupyter

# Setup env
RUN apt-get update
RUN apt-get install -y \
    git wget curl \
    vim \
    tmux \
    htop \
    libsm6 libxext6 libxrender-dev \
    protobuf-compiler python-pil python-lxml

RUN pip install opencv-python==3.4.7.28
RUN pip install tqdm
RUN pip install pillow
RUN pip install lxml

# setup object detect api custom
WORKDIR /tf
RUN git clone https://github.com/Minda93/object_detection_api_custom.git detect_ws

WORKDIR /tf/detect_ws
RUN git checkout ssdlite_mobilenet_v2_fpn_r2_anchor_3

WORKDIR /tf/detect_ws/models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN echo '' >> ~/.bashrc
RUN echo '# for object deetection api custom' >> ~/.bashrc
RUN echo 'export TF_OBJECT_DETECTION="/tf/detect_ws/models/research:/tf/detect_ws/models/research/slim"' >> ~/.bashrc
RUN echo 'export PYTHONPATH="${PYTHONPATH:-${TF_OBJECT_DETECTION}}"' >> ~/.bashrc

WORKDIR /tf/detect_ws
RUN mkdir dataset
RUN mkdir out
RUN mkdir -p save_models/pytorch && mkdir -p save_models/tensorflow

# setup edgetpu
WORKDIR /tf
RUN mkdir install_file
WORKDIR /tf/install_file

# edgetpu python api
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
RUN apt-get install -y python3-edgetpu

# RUN debconf-set-selections | echo 'libedgetpu1-max libedgetpu1-max/Continue_to_install_the_Edge_TPU_runtime_that_runs_at_the_maximum_operating_frequency?_[yes/no] string yes'
RUN echo 'yes' >> apt-get install -f -yqq --no-install-recommends libedgetpu1-max
RUN pip install https://dl.google.com/coral/edgetpu_api/edgetpu-2.14.0-py3-none-any.whl

# check api 
RUN python -c "import edgetpu.basic.edgetpu_utils; print(edgetpu.basic.edgetpu_utils.GetRuntimeVersion())"

# setup edgetpu compiler
RUN git clone https://github.com/google-coral/edgetpu.git
WORKDIR /tf/install_file/edgetpu/compiler/x86_64
RUN cp -r * /usr/local/bin
RUN edgetpu_compiler --version

WORKDIR /tf/detect_ws