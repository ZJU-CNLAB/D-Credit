# D-Credit: A Dynamic Sliding Window of Tensor Communication Mechanism to Accelerate Distributed DNN Training #  
## Introduction ##
This repository contains the codes of the D-Credit paper submitted to IEEE TPDS. D-Credit is a dynamic sliding window of tensor communication mechanism implemented on PyTorch and ByteScheduler frameworks. D-Credit outperforms the state-of-the-art communication scheduling mechanisms ByteScheduler and WFBP.  
<div align=center><img src="system%20architecture.png" width="500"/></div> 

## Installation ##
### Prerequisites ###
We highly recommend using Docker images for experimenting. The following prerequisites shoud be installed in order to use the Docker images for this repository:  
* Ubuntu 18.04  
* CUDA >= 9.0  
* Docker 20.10.18  
* NVIDIA docker
### Data Processing ###
You can unzip the Cifar100 and ImageNet2012 in /data folder and run the following scripts in Python3 to prepare the dataset:  
```
python process_cifar100.py  
python process_imagenet2012.py  
```
### Quick Start ###
You can download this code to /root/code folder and run the following scripts:  
```
cd /root/code/D-Credit/docker  
docker build -t d-credit-pytorch:v1 --no-cache -f d_credit_pytorch.Dockerfile .  
nvidia-docker run -it --net=host --shm-size=32768m -v /data:/data -v /root/code:/root/code d-credit-pytorch:v1 bash  
cd /root/code/D-Credit/docker  
./build-docker-images.sh  
cd /root/code/D-Credit  
chmod 777 ./dist.sh  
dnn=vgg16 nworkers=4 ./dist.sh
```  
Assume that you have 4 GPUs on a single node and everything works well, you will see that there are 4 workers running at a single node training the VGG16 model with the ImageNet2012 dataset using the D-Credit mechanism. The partition size is obtained by Bayesian optimization, and you can tune it manually according to [ByteScheduler's communication scheduling](https://github.com/bytedance/byteps/blob/bytescheduler/bytescheduler/docs/scheduling.md).
