# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import torch
import numpy as np
import argparse, os, sys
import settings
import utils
import logging

from dl_trainer import DLTrainer, _support_datasets, _support_dnns
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import horovod.torch as hvd
from profiling import benchmark
import timeit
import math
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
writer = None

from horovod.torch.mpi_ops import allreduce_async_
from horovod.torch.mpi_ops import allgather_async
from horovod.torch.mpi_ops import broadcast_async_
from horovod.torch.mpi_ops import synchronize
from horovod.torch.mpi_ops import size, local_size, rank, local_rank
from horovod.torch.mpi_ops import init, broadcast
from profiling import CommunicationProfiler
from sklearn.linear_model import LinearRegression

from settings import logger, formatter

os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
# os.environ['HOROVOD_CACHE_CAPACITY'] = '0'
# os.environ['HOROVOD_MPI_THREADS_DISABLE'] = '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="6,7"


def writeFile_add(filename, data):
    file_handle = open(filename, mode='a')
    file_handle.write(data)
    file_handle.close()

def Get_Credit(a, b, S_p, L, M, m, m_re, Delta_t):
    Sc = 0
    Delta_Sc = 0
    for l in range(L):
        if m_re[l]!=0 and Delta_t <= 0 and Delta_t > 0-a:
            Sc = 1
            if m_re[l] == 1:
                m_re[l] = 0
                Delta_Sc = Delta_Sc + m[l] - M[l] / S_p
            else:
                m_re[l] = m_re[l] - 1
                Delta_Sc = 0
            break

        if m_re[l]!=0 and Delta_t>0:
            if (m_re[l]-1)*b*S_p >= Delta_t:
                Sc = Sc + math.ceil(Delta_t/(b*S_p))
                m_re[l] = m_re[l] - math.ceil(Delta_t/(b*S_p))
                break
            else:
                Sc = Sc + m_re[l]
                m_re[l] = 0
                Delta_t = Delta_t - (m_re[l]-1+M[l]/S_p-math.floor(M[l]/S_p))*b*S_p
                Delta_Sc = Delta_Sc + m[l] - M[l]/S_p
    return Sc, Delta_Sc

def Tar(a, b, S_p, Delta_m):
    if Delta_m == 0:
        return 0
    else:
        return a + b*S_p*Delta_m

def D_Credit(a, b, S_p, L, M, t_b):
    delta_1 = L-1
    delta_2 = L
    S_c = np.full(2*L-1, 0.5)
    m = np.full(L, 0.5)
    m_re = np.full(L, 0.5)
    tau_b = np.full(L, 0.5)
    tau_c = np.full(L, 0.5)

    for l in range(L):
        m[l] = math.ceil(M[l]/S_p)
        m_re[l] = 0
    # print('m:', m)

    tau_b[L-1] = 0.0
    for l in range(L-2, -1, -1):
        tau_b[l] = tau_b[l+1] + t_b[l+1]
    # print('tau_b:', tau_b)

    tau_c[L-1] = tau_b[L-1] + t_b[L-1]
    m_re[L-1] = m_re[L-1] + m[L-1]
    for l in range(L-2, -1, -1):
        Delta_t = tau_b[l] + t_b[l] - tau_c[l+1] - a
        # print('Delta_t:', Delta_t)
        S_c[L-1-l-1], Delta_Sc = Get_Credit(a, b, S_p, L, M, m, m_re, Delta_t)
        tau_c[l] = max(tau_c[l+1] + Tar(a, b, S_p, S_c[L-1-l-1]-Delta_Sc), tau_b[l] + t_b[l])
        m_re[l] = m_re[l] + m[l]

    for l in range(L):
        S_c[delta_1 + l] = m_re[l]

    return S_c

def Get_a_and_b(nworkers, S_p):
    if nworkers == 4 and S_p == 1000000.0:
        a = 410.8*1e-06 # s
        b = 1.03e-09
        a = 422.7 * 1e-06  # s
        b = 3.02e-10
    elif nworkers == 4 and S_p == 4000000.0:
        a = 422.7*1e-06 # s
        b = 3.02e-10
        # a = 25e-05 # s
        # b = 2.0589360830118177e-10
    elif nworkers == 4 and S_p == 8000000.0:
        a = 25e-05 # s
        b = 2.0589360830118177e-10
    elif nworkers == 2 and S_p == 4000000.0:
        a = 203.4*1e-06 # s
        b = 1.08e-10
    else:
        a = 422.7*1e-06 # s
        b = 3.02e-10/1.5*2*(nworkers-1)/nworkers
    return a, b

def Benchmarking_communication_performance():
    logger.info('Benchmarking communication performance...')
    comm_profiler = CommunicationProfiler(allreduce_async_, synchronize)
    sizes, times = comm_profiler.benchmark(num_iters=10)

    def _fit_linear_function(x, y):
        X = np.array(x).reshape((-1, 1)) * 4
        Y = np.array(y)
        model = LinearRegression()
        model.fit(X, Y)
        alpha = model.intercept_
        beta = model.coef_[0]
        return alpha, beta/4000000

    alpha, beta = _fit_linear_function(sizes, times)
    # self.alpha = alpha
    # self.beta = beta
    alpha_tensor = torch.ones(1) * alpha
    beta_tensor = torch.ones(1) * beta
    alpha_tensor = broadcast(alpha_tensor, root_rank=0)
    beta_tensor = broadcast(beta_tensor, root_rank=0)
    # if rank() != 0:
    #     self.alpha = float(alpha_tensor[0])
    #     self.beta = float(beta_tensor[0])
    logger.info('Communication performance fitted with f(p)=a+b*p, where a={} and b={}'.format(alpha, beta))
    res = 'alpha {:.10f} beta {:.10f}'.format(alpha, beta) + '\n'
    filename = './result/D-Credit-alpha_and_beta.txt'
    writeFile_add(filename, res)
    print("sizes:", sizes, '\n', 'times:', times, '\n')
    return alpha, beta

def run(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compression, partition):
    rank = hvd.rank()
    torch.cuda.set_device(rank%nwpernode)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix='allreduce', pretrain=pretrain, num_steps=num_steps, tb_writer=writer)

    init_epoch = torch.ones(1) * trainer.get_train_epoch()
    init_iter = torch.ones(1) * trainer.get_train_iter()
    trainer.set_train_epoch(int(hvd.broadcast(init_epoch, root_rank=0)[0]))
    trainer.set_train_iter(int(hvd.broadcast(init_iter, root_rank=0)[0]))

    seq_layernames, layerwise_times, layerwise_sizes = benchmark(trainer)
    layerwise_times = comm.bcast(layerwise_times, root=0)
    if rank == 0:
        logger.info('layerwise backward times: %s', list(layerwise_times))
        logger.info('layerwise backward sizes: %s', list(layerwise_sizes))
    logger.info('Bencharmked backward time: %f', np.sum(layerwise_times))
    logger.info('Model size: %d', np.sum(layerwise_sizes))

    # D-Credit
    model = trainer.net
    S_p = int(os.environ.get('BYTESCHEDULER_PARTITION', 1000000))
    a, b = Get_a_and_b(nworkers, S_p)
    L = len(list(model.named_parameters()))
    logger.info('L: %d', L)
    M = []
    t_b = []
    for l in range(L):
        M.append(float(layerwise_sizes[l]))
        t_b.append((float(layerwise_times[l])))
    S_c = D_Credit(a, b, S_p, L, M, t_b)
    logger.info('Credit size: %s', list(S_c))


    # bytescheduler wrapper
    use_bytescheduler_and_D_Credit = int(os.environ.get('USE_BYTESCHEDULER', '0'))
    if use_bytescheduler_and_D_Credit > 0:
        if partition:
            os.environ["BYTESCHEDULER_PARTITION"] = str(1000 * partition)
        import bytescheduler.pytorch.horovod as bsc
        bsc.init()

    optimizer = hvd.DistributedOptimizer(trainer.optimizer, named_parameters=trainer.net.named_parameters(), compression=compression)

    iters_per_epoch = trainer.get_num_of_training_samples() // (nworkers * batch_size * nsteps_update)
    if use_bytescheduler_and_D_Credit > 0:
        optimizer = bsc.ScheduledOptimizer(model, optimizer, max_epochs * iters_per_epoch, S_c)

    hvd.broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    trainer.update_optimizer(optimizer)

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 10 if iters_per_epoch > 10 else iters_per_epoch-1

    display_speed = 100 if iters_per_epoch > 100 else iters_per_epoch - 1
    accumulate_iters = 400
    img_secs = []

    for epoch in range(max_epochs):
        for i in range(iters_per_epoch):
            s = time.time()
            optimizer.zero_grad()
            for j in range(nsteps_update):
                if j < nsteps_update - 1 and nsteps_update > 1:
                    optimizer.local = True
                else:
                    optimizer.local = False
                trainer.train(1)
            trainer.update_model()
            times.append(time.time()-s)

            img_secs.append(batch_size * nsteps_update / (time.time()-s))
            display_speed -= 1
            if display_speed == 0:
                res = 'Total img/sec on {} GPU(s): {:.2f}'.format(nworkers, nworkers * np.mean(img_secs)) + '\n'
                filename = './result/D-Credit-'+dnn+'-'+str(nworkers)+'-'+str(batch_size)+'.txt'
                writeFile_add(filename, res)
                img_secs = []
                display_speed = 100 if iters_per_epoch > 100 else iters_per_epoch - 1

            if i % display == 0 and i > 0:
                time_per_iter = np.mean(times)
                logger.warn('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
            accumulate_iters -= 1
            if accumulate_iters == 0:
                os._exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AllReduce trainer")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nsteps-update', type=int, default=1)
    parser.add_argument('--nworkers', type=int, default=1, help='Just for experiments, and it cannot be used in production')
    parser.add_argument('--nwpernode', type=int, default=1, help='Number of workers per node')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=_support_datasets, help='Specify the dataset for training')
    parser.add_argument('--dnn', type=str, default='resnet50', choices=_support_dnns, help='Specify the neural network for training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Specify the data root path')
    parser.add_argument('--saved-dir', type=str, default='.', help='Specify the saved weights or gradients root path')
    parser.add_argument('--lr', type=float, default=0.1, help='Default learning rate')
    parser.add_argument('--max-epochs', type=int, default=settings.MAX_EPOCHS, help='Default maximum epochs to train')
    parser.add_argument('--pretrain', type=str, default=None, help='Specify the pretrain path')
    parser.add_argument('--num-steps', type=int, default=35)
    parser.add_argument('--fp16-allreduce', action='store_true', default=False, help='use fp16 compression during allreduce')
    parser.add_argument('--partition', type=int, default=None, help='partition size')
    args = parser.parse_args()
    batch_size = args.batch_size * args.nsteps_update
    logdir = '%s-n%d-bs%d-lr%.4f' % (args.dnn, args.nworkers, batch_size, args.lr)
    relative_path = './logs/%s'%logdir
    gradient_relative_path = None 
    utils.create_path(relative_path)
    rank = 0
    if args.nworkers > 1:
        hvd.init()
        rank = hvd.rank()
    if rank == 0:
        tb_runs = './runs/%s'%logdir
        writer = None #SummaryWriter(tb_runs)
    logfile = os.path.join(relative_path, settings.hostname+'-'+str(rank)+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)

    # Benchmarking communication performance
    # a, b = Benchmarking_communication_performance()
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    run(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, compression, args.partition)
