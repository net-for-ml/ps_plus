import torch
import torch.nn as nn
import torch.optim as optim

class ps():
    def __init__(self, model, optimizer, ParameterServer, ps, rank):
        super(ps, self).__init__(model, optimizer, ParameterServer, ps, rank)

    def push_and_pull(self):
        dw_gpu = self.optimizer.get_delta_weight()
        dw_cpu = [item.to('cpu') for item in dw_gpu]
                
        fut0 = remote_method(self.ParameterServer.update_and_fetch, self.ps[0], self.rank, dw_cpu[:80])
        fut1 = remote_method(self.ParameterServer.update_and_fetch, self.ps[1], self.rank, dw_cpu[80:])
        cur_steps0, new_params0 = fut0.wait()
        cur_steps2, new_params1 = fut1.wait()
        cur_steps = cur_steps0 + cur_steps1
        new_params = new_params0 + new_params1
        np_gpu = [item.to(device) for item in new_params]
        with torch.no_grad():
            for param, value in zip(self.model.parameters(), np_gpu):
                param.copy_(value)

        return cur_steps


class ps_plus():
    def __init__(self, model, optimizer, ParameterServer, ps, fut, rank):
        super(ps_plus, self).__init__(model, optimizer, ParameterServer, ps, fut, rank)

    def pull_and_push(self):
        dw_gpu = self.optimizer.get_delta_weight()
        dw_cpu = [item.to('cpu') for item in dw_gpu]
                
        cur_steps1, new_params0 = self.fut[0].wait()
        cur_steps2, new_params1 = self.fut[1].wait()
        cur_steps = cur_steps0 + cur_steps1
        new_params = new_params0 + new_params1
        fut0 = remote_method(self.ParameterServer.update_and_fetch, self.ps[0], self.rank, dw_cpu[:80])
        fut1 = remote_method(self.ParameterServer.update_and_fetch, self.ps[1], self.rank, dw_cpu[80:])
        np_gpu = [item.to(device) for item in new_params]
        with torch.no_grad():
            for param, value, grad in zip(self.model.parameters(), np_gpu, dw_gpu):
                param.copy_(value-grad)

        return cur_steps, [fut0, fut1]


class optim_SGD(optim.SGD):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(optim_SGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad_param = param.grad
                if weight_decay != 0:
                    grad_param = grad_param.add(param, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[param]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad_param).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad_param, alpha=1-dampening)
                    if nesterov:
                        grad_param = grad_param.add(buf, alpha=momentum)
                    else:
                        grad_param = buf
                
                param.add_(grad_param, alpha=-group['lr'])
        
        return loss
    
    @torch.no_grad()
    def get_delta_weight(self):
        delta_weight = []
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for param in group['params']:       
                if param.grad is None:
                    continue
                
                grad_param = param.grad
                if weight_decay != 0:
                    grad_param = grad_param.add(param, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[param]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad_param).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad_param, alpha=1-dampening)
                    if nesterov:
                        grad_param = grad_param.add(buf, alpha=momentum)
                    else:
                        grad_param = buf
                
                delta_weight.append(group['lr']*grad_param)
        
        return delta_weight
