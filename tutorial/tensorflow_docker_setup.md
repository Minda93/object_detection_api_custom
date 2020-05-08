# tensorflow docker 環境建置

## Environment

| tool                | version                   |
|---------------------|---------------------------|
| nvidia-driver       | >= 430.64                 |
| opencv-python       | 3.4.7.28                  |
| edgetpu_compiler    | 2.1.302470888             |
| edgetpu_runtime     | 13                        |
| pip                 | >= 19.3.1                 |
| protobuf-compiler   | 3.0.0                     |
| python-pil          | 7.1.1                     |
| python-lxml         | 4.5.0                     |
| tqdm                | 4.45.0                    |

## Run Container

```bash
  # jupyter gpu version
  $ sudo docker pull tensorflow/tensorflow:1.15.2-gpu-jupyter
   
  # check images id
  $ sudo docker images
  
  # run tensorflow container
  $ sudo docker run -it --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0 -p 8888:8888 -name <name> -v <share_folder>:/tf/<share_folder> -shm-size='64g' <image_name or image_id>
  
  # 可能有更好的方法
  # check jupyter key
  $ sudo docker attach <container_name>
  # use ctrl + c (you can see jupyter key)
  
  # 開啟瀏覽器
  # input ip:port into jupyter
  # input jupyter and change password and enter jupyter
  
  # set new password
  $ sudo docker restert  <container_name>
  
```

## Setup env
  
  * you can write dockerfile to build custom image

1. setup env
```bash
  # install python-opencv 
  
  $ apt-get update
  $ apt-get install -y libsm6 libxext6 libxrender-dev
  $ pip install opencv-python==3.4.7.28 --user
  
  # install some packages
  $ apt-get install vim
  $ apt-get install protobuf-compiler python-pil python-lxml
  
  $ pip install tqdm
  $ pip install pillow
  $ pip install lxml
  
```
2. setup edgetpu env

```bash
  # add edgetpu key
  $ echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
  $ curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  $ sudo apt-get update
  $ sudo apt-get install python3-edgetpu

  # the edge TPU runtime, at maximum operating frequency.
  $ sudo apt-get install  libedgetpu1-max
  
  # ======================
  
  # install python api
  $ pip3 install https://dl.google.com/coral/edgetpu_api/edgetpu-2.13.0-py3-none-any.whl

  # check edgetpu version
  $ python3 -c "import edgetpu.basic.edgetpu_utils; print(edgetpu.basic.edgetpu_utils.GetRuntimeVersion())"

  # you maybe see below detail
  # BuildLabel(COMPILER=5.4.0 20160609,DATE=redacted,TIME=redacted,CL_NUMBER=291256449), RuntimeVersion(13)
  
  # ======================
  # install edgetpu compiler
  $ git clone https://github.com/google-coral/edgetpu.git

  # x86_64
  $ cd edgetpu/compiler/x86_64
  # aarch64
  $ cd edgetpu/compiler/aarch64

  # maybe path (choice)
  $ sudo cp -r * /usr/local/bin
  $ sudo cp -r * /usr/bin
```

## Reference
   
  * [tensorflow docker run](https://qiita.com/hrappuccino/items/fe76e2ed014c16171e47)
  * [coral edgetpu](https://coral.ai/software/#debian-packages)
  * [edgetpu github](https://github.com/google-coral/edgetpu)
