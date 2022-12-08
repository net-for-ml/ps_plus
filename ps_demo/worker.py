import argparse
import math
import os
import sys
import time
from collections import OrderedDict
from ctypes import c_bool
from multiprocessing import Process, Value
from multiprocessing.managers import BaseManager

import numpy as np
import torch
import torch.distributed as dist
from utils.divide_data import partition_dataset, select_dataset, DataPartitioner
from utils.models import MnistCNN, AlexNetForCIFAR
from utils.utils_data import get_data_transform
from utils.utils_model import MySGD, test_model
from torch.autograd import Variable
from torch.multiprocessing import Process as TorchProcess
from torchvision import datasets

parser = argparse.ArgumentParser()
# The basic configuration of the cluster
parser.add_argument('--ps_ip', type=str, default='127.0.0.1')
parser.add_argument('--ps_port', type=str, default='29500')
parser.add_argument('--this_rank', type=int, default=1)
parser.add_argument('--all_workers', type=str, default='1-2')

# The configuration of model and dataset
parser.add_argument('--data_dir', type=str, default='../../data')
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--model', type=str, default='MnistCNN')

# The configuration of different hyper-parameters for training
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_bsz', type=int, default=100)
parser.add_argument('--heterogeneity', type=float, default=1.0)
parser.add_argument('--hetero_allocation', type=str, default='1.0-1.0')

args = parser.parse_args()


def unbalanced_partition_dataset(dataset, hetero):
    """ Partitioning Data """
    computing_capacity = [float(v) for v in hetero.split('-')]
    norm_factor = sum(computing_capacity)
    partition_sizes = [v/norm_factor for v in computing_capacity]
    partition = DataPartitioner(dataset, partition_sizes)
    return partition

def run(rank, model, train_data, test_data, queue, param_q, stop_flag):
    # Fetch the initial parameters from the server side (we called it parameter_server)
    while True:
        if not param_q.empty():
            param_dict = param_q.get()
            tmp = OrderedDict(map(lambda item: (item[0], torch.from_numpy(item[1])),
                                  param_dict.items()))
            model.load_state_dict(tmp)
            break
    print('Model recved successfully!')

    if args.model == 'MnistCNN':
        optimizer = MySGD(model.parameters(), lr=0.01, momentum=0.5)
        criterion = torch.nn.NLLLoss()
    else:
        optimizer = MySGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        criterion = torch.nn.NLLLoss()

    print('Begin!')


    for epoch in range(int(args.epochs)):
        model.train()
        epoch_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_data):
            it_start = time.time()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            delta_ws = optimizer.get_delta_w()

            it_end = time.time()
            it_duration = it_end - it_start
            # Mimic the heterogeneity through delay
            # The "heterogeneity" is a ratio of the maximum computing capacity
            sleep_time = (1.0 / args.heterogeneity - 1.0) * it_duration
            time.sleep(sleep_time)

            try:

                if delta_ws:
                    queue.put({
                        rank: [[v.numpy() for v in delta_ws],loss.data.numpy(), np.array(args.batch_size), False]
                    })

                for idx, param in enumerate(model.parameters()):
                    tmp_tensor = torch.zeros_like(param.data)
                    dist.recv(tensor= tmp_tensor, src=0)
                    param.data = tmp_tensor

            except Exception as e:
                print(str(e))
                print('Should Stop: {}!'.format(stop_flag.value))
                break

        # Check the top 1 test accuracy after training
        print("test Model:",epoch)
        test_model(rank, model, test_data, criterion=criterion)
        if stop_flag.value:
            break
    queue.put({rank: [[], [], [], True]})
    print("Worker {} has completed epoch {}!".format(args.this_rank, epoch))


def init_processes(rank, size, model,
                   train_dataset, test_dataset,
                   q, param_q, stop_flag,
                   fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = args.ps_ip
    os.environ['MASTER_PORT'] = args.ps_port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, model, train_dataset, test_dataset, q, param_q, stop_flag)


def capture_stop(stop_signal, flag: Value):
    while True:
        if not stop_signal.empty():
            flag.value = True
            print('Time Up! Stop: {}!'.format(flag.value))
            break


if __name__ == "__main__":

    if args.model == 'MnistCNN':
        model = MnistCNN()
        train_t, test_t = get_data_transform('mnist')
        train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                       transform=train_t)
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                      transform=test_t)

        train_bsz = args.batch_size
        test_bsz = args.test_bsz

    elif args.model == 'AlexNet':
        model = AlexNetForCIFAR()
        train_t, test_t = get_data_transform('cifar')
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                         transform=train_t)
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                        transform=test_t)

        train_bsz = args.batch_size
        test_bsz = args.test_bsz

    else:
        print('Model must be {} or {}!'.format('MnistCNN', 'AlexNet'))
        sys.exit(-1)

    workers = [int(v) for v in str(args.all_workers).split('-')]
    train_data = unbalanced_partition_dataset(train_dataset, args.hetero_allocation)
    test_data = partition_dataset(test_dataset, workers)

    this_rank = args.this_rank
    train_data = select_dataset(workers, this_rank, train_data, batch_size=train_bsz)
    test_data = select_dataset(workers, this_rank, test_data, batch_size=test_bsz)

    class MyManager(BaseManager):
        pass

    MyManager.register('get_queue')
    MyManager.register('get_param')
    MyManager.register('get_stop_signal')
    manager = MyManager(address=(args.ps_ip, 5000), authkey=b'queue')
    manager.connect()

    q = manager.get_queue()
    param_q = manager.get_param()
    stop_signal = manager.get_stop_signal()

    stop_flag = Value(c_bool, False)
    stop_p = Process(target=capture_stop, args=(stop_signal, stop_flag))

    p = TorchProcess(target=init_processes, args=(this_rank, len(workers)+1,
                                                  model,
                                                  train_data, test_data,
                                                  q, param_q, stop_flag,
                                                  run))
    p.start()
    stop_p.start()
    p.join()
    stop_p.join()
