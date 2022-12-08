import argparse
import os
import time
import random
import numpy as np
from threading import Lock

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torchvision import datasets, transforms

from utils.optims import ps
from utils.optims import ps_plus
from utils.optims import optim_SGD
from utils.models import ResNet50_CIFAR10


num_epoch = 300
param_server = None
global_lock = Lock()


# ---------- Helper Methods ----------
def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), call_method, args=args, kwargs=kwargs)


# ---------- Parameter Server ----------
class ParameterServer(nn.Module):
    def __init__(self, **kwargs):
        super(ParameterServer, self).__init__()
        self.model = ResNet50_CIFAR10()
        self.max_stale = kwargs['max_stale']
        self.cur_steps = [0]*kwargs['num_worker']
        self.min_step = 0
        self.lock = Lock()
        self.params = []
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = False
            if (self.rank==0 and i<80) or (self.rank==1 and i>=80):
                self.params.append(param)
    
    def get_params(self):
        return self.cur_steps, self.params
    
    def update_and_fetch(self, rank, grads):
        with self.lock:
            self.cur_steps[rank-1] += 1
            self.min_step = min(self.cur_steps)
            for param, grad in zip(self.params, grads):
                param -= 0.125*grad

        cur_step = self.cur_steps[rank-2]
        while cur_step > self.min_step + self.max_stale:
            print('Worker{}: {}; Progress: {}'.format(rank, cur_step, self.min_step))
            time.sleep(0.1)
        return self.cur_steps, self.params


def get_param_server(**kwargs):
    global param_server
    with global_lock:
        if not param_server:
            param_server = ParameterServer(**kwargs)
        return param_server


def run_server(**kwargs):
    rank = kwargs['this_rank']
    world_size = kwargs['world_size']
    server_name = 'server{}'.format(rank)
    rpc.init_rpc(server_name, rank=rank, world_size=world_size)
    print('RPC initialized! Running server{}.'.format(rank))
    
    rpc.shutdown()
    print('RPC shutdown on server{}.'.format(rank))


def get_accuracy(model, test_loader, device):
    model.eval()
    
    num_samples = 0
    num_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = output.max(1)
            num_samples += pred.size(0)
            num_correct += (pred == target).sum()
    return float(num_correct) / num_samples


def run_worker(**kwargs):
    # Setup the process on worker
    rank = kwargs['this_rank']
    world_size = kwargs['world_size']
    worker_name = 'worker{}'.format(rank)
    rpc.init_rpc(worker_name, rank=rank, world_size=world_size)
    print('RPC initialized! Running worker{}.'.format(rank))
    
    ps0 = rpc.remote('server0', get_param_server, kwargs=kwargs)
    ps1 = rpc.remote('server1', get_param_server, kwargs=kwargs)
    ps = [ps0, ps1]
    
    device = torch.device('cuda:{}'.format(rank-1))
    model = ResNet50_CIFAR10().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim_SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
    
    # Load train and test datasets
    print('Load train and test datasets ...')
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, world_size-1, rank-1)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=250, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=250)
    
    # Start the training process
    print('Start the training process with {} ...'.format(kwargs['train_mode']))
    
    accuracy = []
    comp_time = 0
    comm_time = 0
    tik = time.time()
    if kwargs['train_mode'] == 'NEW':
        fut0 = remote_method(ParameterServer.get_params, ps[0])
        fut1 = remote_method(ParameterServer.get_params, ps[1])
        fut = [fut0, fut1]
        for epoch in range(num_epoch):
            train_sampler.set_epoch(epoch)
            model.train()
            for i, (data, target) in enumerate(train_loader):
                s_time = time.time()
                
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                e_time = time.time()
                comp_time = comp_time + e_time - s_time
                s_time = time.time()

                cur_steps, fut = ps_plus.pull_and_push(model, optimizer, ParameterServer, ps, fut, rank)
                
                e_time = time.time()
                comm_time = comm_time + e_time - s_time
            
            acc = get_accuracy(model, test_loader, device)
            accuracy.append(acc)
            print('Rank {} -- Epoch {}: {:0.2f}%'.format(rank, epoch+1, 100*acc))
            # lr_scheduler(cur_steps) goes here
    else:
        for epoch in range(num_epoch):
            train_sampler.set_epoch(epoch)
            model.train()
            for i, (data, target) in enumerate(train_loader):
                s_time = time.time()
                
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                e_time = time.time()
                comp_time = comp_time + e_time - s_time
                s_time = time.time()
                
                cur_steps = ps.push_and_pull()
                
                e_time = time.time()
                comm_time = comm_time + e_time - s_time
            
            acc = get_accuracy(model, test_loader, device)
            accuracy.append(acc)
            print('Rank {} -- Epoch {}: {:0.2f}%'.format(rank, epoch+1, 100*acc))
            # lr_scheduler(cur_steps) goes here
    
    tok = time.time()
    print('Total run time: {}'.format(tok-tik))
    print('Computation time: {}'.format(comp_time))
    print('Communication time: {}'.format(comm_time))
    
    rpc.shutdown()
    print('RPC shutdown on worker{}.'.format(rank))


# ---------- Launcher Function ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ParameterServer with RPC')
    
    parser.add_argument('--ps_addr', type=str, default='192.168.149.22')
    parser.add_argument('--ps_port', type=str, default='23333')
    parser.add_argument('--this_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=10)
    parser.add_argument('--num_server', type=int, default=2)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--train_mode', type=int, default=3)
    parser.add_argument('--max_stale', type=int, default=100000)
    
    args = parser.parse_args()
    
    torch.manual_seed(23)
    os.environ['MASTER_ADDR'] = args.ps_addr
    os.environ['MASTER_PORT'] = args.ps_port
    
    max_stale = args.max_stale
    if args.train_mode == 0:
        max_stale = 0
    if args.train_mode == 1:
        max_stale = 100000
    
    train_mode_list = ['BSP', 'ASP', 'SSP', 'NEW']
    train_mode = train_mode_list[args.train_mode]
    
    kwargs_server = {'this_rank':args.this_rank, 
                     'world_size':args.world_size}
    
    kwargs_worker = {'this_rank':args.this_rank, 
                     'world_size':args.world_size, 
                     'num_server':args.num_server, 
                     'num_worker':args.num_worker, 
                     'train_mode':train_mode, 
                     'max_stale':max_stale}
    
    if args.this_rank < args.num_server:
        p = mp.Process(target=run_server, kwargs=kwargs_server)
    else:
        p = mp.Process(target=run_worker, kwargs=kwargs_worker)
    
    p.start()
    p.join()
