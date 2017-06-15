# About
This repository is a collection of commonly used Keras examples adapted to be used for profiling the performance of different Keras backends, mainly Apache MXNet (You could change the backend to TensorFlow). These performance profiling data may not be fully accurate. However, this should help in giving a relative insights into various performance metrics with different Keras backends.

# Installation
# Ubuntu
# GPU

## Setup NVIDIA CUDA and cuDNN

Install the following NVIDIA libraries on your GPU machine:

1. Install CUDA 8.0 following the NVIDIA's [installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
2. Install cuDNN 5 for CUDA 8.0 following the NVIDIA's [installation guide](https://developer.nvidia.com/cudnn). You need to register with NVIDIA for downloading the cuDNN library.

**Note:** Make sure to add CUDA install path to `LD_LIBRARY_PATH`.

For Example, if you have downloaded CUDA debian package (`cuda-repo-ubuntu1604_8.0.61-1_amd64.deb`) and cuDNN 5.1 library (`cudnn-8.0-linux-x64-v5.1.tgz`), below are set of commands you run to setup CUDA and cuDNN.

```bash

#  Setup CUDA 8.0.
$  sudo apt-get update
$  sudo apt-get install build-essential
$  sudo apt-get install linux-headers-$(uname -r)
#  Assuming you have downloaded CUDA deb package from https://developer.nvidia.com/cuda-downloads
$  sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
$  sudo apt-get update
$  sudo apt-get install cuda

$  export CUDA_HOME=/usr/local/cuda-8.0
$  PATH=${CUDA_HOME}/bin:${PATH}
$  export PATH
$  export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

#  Setup cuDNN 5.1 for CUDA 8.0.
#  Assuming you have registered with NVIDA and downloaded cuDNN 5.1 for CUDA 8 from https://developer.nvidia.com/cudnn
$  tar -xvzf cudnn-8.0-linux-x64-v5.1.tgz
$  sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
$  sudo cp cuda/lib64/* /usr/local/cuda/lib64/
```

You can verify your CUDA setup with following commands.

```bash
$  nvcc --version
$  nvidia-smi
```

## Install Prerequisites

Install Prerequisites with following commands.

```bash
#  Install git
$  sudo apt-get update
$  sudo apt-get install -y build-essential git
#  Install pip
$  sudo apt-get install -y wget python
$  wget https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py$ sudo pip install numpy
#  Dependencies required by Keras
$  sudo pip install nose
$  sudo pip install nose-parameterized
#  Dependencies required for profiling
$  sudo pip install memory_profiler
```

## Install Apache MXNet

Install latest MXNet from source. Below instructions derived from [MXNet install guide](http://mxnet.io/get_started/install.html).

```bash
#  BLAS library
$  sudo apt-get install -y libopenblas-dev
#  OpenCV
$  sudo apt-get install -y libopencv-dev
#  Download MXNet sources and build MXNet core shared library
$  git clone --recursive https://github.com/dmlc/mxnet
$  cd mxnet
$  make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1

#  Install MXNet Python bindings
$  sudo apt-get install -y python-dev python-setuptools python-numpy
$  cd python
$  sudo python setup.py install
```

## Install TensorFlow for GPU

```bash
$  sudo pip install tensorflow-gpu
```
Reference - [TensorFlow install guide](https://www.tensorflow.org/install/install_linux#install_tensorflow).

## Install Keras 1.2.2 with MXNet backend

DMLC's fork of Keras has MXNet backend. For more details about known issues, not supported functionalities [see here](https://github.com/dmlc/keras)

```bash
$  git clone https://github.com/dmlc/keras.git ~/keras --recursive
$  cd ~/keras
$  sudo python setup.py install
```

# How to run

First, you clone this repository to your test machine.

```bash
$  git clone --recursive https://github.com/sandeep-krishnamurthy/keras-mxnet-benchmarks
```

You use 3 environment variables - `KERAS_BACKEND`, `MXNET_KERAS_TEST_MACHINE` and `GPU_NUM` to control the backend and machine set up for running the profiling.

Below are allowed values for each of these environment variables.

1. KERAS_BACKEND - "mxnet" or "tensorflow".
2. MXNET_KERAS_TEST_MACHINE = "CPU" or "GPU".
3. GPU_NUM = Integer. Number of GPUs to be used for profiling. Ex: GPU_NUM = 8, will use 8 GPUs.

To run RESTNET50 profiling with MXNet backend using 4 GPUs, you run the following commands.

First, set the appropriate environment variables.

```bash
#  Change "mxnet" to "tensorflow" for profiling with TensorFlow backend.
$  export KERAS_BACKEND=mxnet
$  export MXNET_KERAS_TEST_MACHINE=GPU
$  export GPU_NUM=4
```

Run the profiling python script. You will the profiling results on the terminal.

```bash
$  python keras-mxnet-benchmarks/keras1.2/test_cifar_resnet.py
```

# Credits/References
1. Resnet50 Keras code is borrowed from - [raghakot/keras-resnet](https://github.com/raghakot/keras-resnet/blob/master/resnet.py)
2. Most of the examples are borrowed from - [fchollet/keras/examples](https://github.com/fchollet/keras/blob/master/examples/)
