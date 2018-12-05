import torch
from torch import nn
from torch.autograd import Variable

class loss_scaler(object):

    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

    def backward(self, loss):
        scaled_loss = loss * self.scale_factor
        scaled_loss.backward()

class mixed_precision_optimizer(object):

    def __init__(self, init_optimizer, scale_factor=1.0):
        
        self.optimizer = init_optimizer
        self.scale_factor=scale_factor
        self.fp16 = []
        self.fp32fromfp16 = []
        self.fp32fromfp32 = []

        for i, param_g in enumerate(self.optimizer.param_groups):
            tmp_fp16 = []
            tmp_fp32 = []
            tmp_fp32fromfp16 = []
            for i, param in enumerate(param_g['params']):
                if param.requires_grad:
                    if param.type() == "torch.cuda.HalfTensor":
                        tmp_fp16.append(param)
                        master_param = param.detach().clone().float()
                        master_param.requires_grad = True
                        param_g['params'][i] = master_param
                        tmp_fp32fromfp16.append(master_param)
                        if param in self.optimizer.state:
                            self.optimizer.state[master_param] = self.optimizer.state.pop(param)
                    elif param.type() == "torch.cuda.FloatTensor":
                        tmp_fp32.append(param)
                        param_g['params'][i] = param
                    else:
                        print ("miaomiaomiao: wrong type of parameters")
            self.fp16.append(tmp_fp16)
            self.fp32fromfp32.append(tmp_fp32)
            self.fp32fromfp16.append(tmp_fp32fromfp16)
        
        self.optimizer.load_state_dict(self.optimizer.state_dict())

        self.ls = loss_scaler(scale_factor)

    def state_dict(self):
        ret = {}
        ret['loss_scaler'] = self.ls
        ret['optimizer_state_dict'] = self.optimizer.state_dict()
        ret['fp32fromfp16'] = self.fp32fromfp16
        return ret

    def load_state_dict(self, state_dict):

        self.ls = state_dict['loss_scaler']
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        for cur_g, saved_g in zip(self.fp32fromfp16, state_dict['fp32fromfp16']):
            for cur, saved in zip(cur_g, saved_g):
                cur.data.copy_(saved.data)

    def _zero_grad(self, param):
        param.grad.detach_()
        param.grad.zero_()

    def zero_grad(self):
        for g in self.optimizer.param_groups:
            for p in g['params']:
                if p.grad is not None:
                    self._zero_grad(p)
        for g in self.fp16:
            for p in g:
                if p.grad is not None:
                    self._zero_grad(p)

    def model_grads2master_grads(self, model_p, master_p):
        for model, master in zip(model_p, master_p):
            if model.grad is not None:
                if master.grad is None:
                    master.grad = Variable(master.data.new(*master.data.size()))
                master.grad.data.copy_(model.grad.data)
            else:
                master.gard = None

    def step(self):
        ret = self.optimizer.step()               
        for fp16_g, fp32fromfp16_g in zip(self.fp16, self.fp32fromfp16):
            self.model_grads2master_grads(fp16_g, fp32fromfp16_g)
        return ret

    def backward(self, loss):
        self.ls.backward(loss.float())
        # update master grads
        for fp16_g, fp32fromfp16_g in zip(self.fp16, self.fp32fromfp16):
            self.model_grads2master_grads(fp16_g, fp32fromfp16_g)
        # Downscale
        if self.scale_factor != 1.0:
            for g in self.optimizer.param_groups:
                for param in g['params']:
                    if param.grad is not None:
                        param.grad.data.mul_(1. / self.scale_factor)
        



















