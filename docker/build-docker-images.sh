#!/bin/bash
# Date：2022-12-02
# Author: Create by Yunqi Gao
# Description: This script function installs ByteScheduler with D-Credit.
# Version： 1.0

cd && cd byteps/bytescheduler/bytescheduler/common && mv /root/code/D-Credit/docker/file/bytecore.py bytecore.py && cd &&

cd byteps/bytescheduler/bytescheduler/pytorch && mv /root/code/D-Credit/docker/file/horovod.py horovod.py &&
mv /root/code/D-Credit/docker/file/horovod_task.py horovod_task.py && cd &&

cd /root/code/D-Credit/byteps/bytescheduler && python setup.py install &&

pip install mpi4py
