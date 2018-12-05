import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class mixed_precision_optimizer(object):

    def __init__(self, init_optimizer, scale_factor=1.0):
        
        self.optimizer = init_optimizer
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

    def load_state_dict(self, state_dict):

    self.ls = state_dict['loss_scaler']
    self.optimizer.load_state_dict(state_dict['optimizer_state_dict']

    for cur_g, saved_g in zip(self.fp32fromfp16, state_dict['fp32fromfp16']):
        for cur, saved in zip(cur_g, saved_g):
            cur.data.copy_(saved.data)

    def step(self):
        self._update_scale()
        ret = self.optimizer.step()               
        self._master_params_to_model_params()

        return ret



















