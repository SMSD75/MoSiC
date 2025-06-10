import torch
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from utils.my_utils import cosine_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

class PatchCorrespondenceOptimizer:
    def __init__(self, model_parameters_dict, init_lr, peak_lr, decay_half_life, warmup_steps, grad_norm_clipping,
                 init_weight_decay, peak_weight_decay, max_iter):
        self.model_parameters_dict = model_parameters_dict
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.decay_half_life = decay_half_life
        self.warmup_steps = warmup_steps
        self.grad_norm_clipping = grad_norm_clipping
        self.init_weight_decay = init_weight_decay
        self.peak_weight_decay = peak_weight_decay
        self.max_iter = max_iter
        
        self.optimizer = None
        self.scheduler = None
        self.current_step = 0
    
    def setup_optimizer(self, optimizer_type='AdamW'):
        if optimizer_type == 'AdamW':
            self.optimizer = AdamW(self.model_parameters_dict, lr=self.init_lr, weight_decay=self.init_weight_decay)
        elif optimizer_type == 'SGD':
            self.optimizer = SGD(self.model_parameters_dict, lr=self.init_lr, weight_decay=self.init_weight_decay)
        else:
            raise ValueError("Unsupported optimizer type. Choose 'Adam' or 'SGD'.")
    
    def setup_scheduler(self):
        if self.decay_half_life > 0:
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.5**(1/self.decay_half_life))
        else:
            self.scheduler = None
    
    def step(self):
        self.adjust_weight_decay()
        self.optimizer.step()
        if self.grad_norm_clipping:
            parameters = []
            for param_dict in self.model_parameters_dict:
                parameters += param_dict['params']
            torch.nn.utils.clip_grad_norm_(parameters, self.grad_norm_clipping)
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            self.warmup_lr()
        else:
            self.update_lr()
    
    def warmup_lr(self):
        if self.current_step < self.warmup_steps:
            warmup_factor = self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.init_lr +  warmup_factor * (self.peak_lr - self.init_lr)
    
    def adjust_weight_decay(self):
        if self.current_step <= self.max_iter:
            weight_decay_factor = self.current_step / self.max_iter
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = self.init_weight_decay + weight_decay_factor * (self.peak_weight_decay - self.init_weight_decay)
    
    def update_lr(self):
        if self.scheduler is not None:
            self.scheduler.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def get_weight_decay(self):
        return self.optimizer.param_groups[0]['weight_decay']


class DINO_VQVAE_Optimizer:
    def __init__(self, optimizer, scheduling, weight_decay, weight_decay_end, max_epochs, train_iters_per_epoch, final_lr):
        self.wd_schedule = cosine_scheduler(weight_decay, weight_decay_end, max_epochs, train_iters_per_epoch)
        self.current_step = 0
        self.optimizer = optimizer
        self.scheduling = scheduling
        self.train_iters_per_epoch = train_iters_per_epoch
        self.max_epochs = max_epochs
        self.final_lr = final_lr

    def step(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["weight_decay"] = self.wd_schedule[self.current_step]
        self.current_step += 1
        self.optimizer.step()
        self.update_lr()
    
    def setup_optimizer(self, optimizer_type='AdamW'):
        if self.optimizer is None:
            if optimizer_type == 'AdamW':
                self.optimizer = AdamW(self.model_parameters, lr=self.init_lr, weight_decay=self.init_weight_decay)
            elif optimizer_type == 'SGD':
                self.optimizer = SGD(self.model_parameters, lr=self.init_lr, weight_decay=self.init_weight_decay)
            else:
                raise ValueError("Unsupported optimizer type. Choose 'Adam' or 'SGD'.")

    
    def setup_scheduler(self):
        if self.scheduling == 'cosine':
            self.schedular = CosineAnnealingLR(self.optimizer, T_max=self.train_iters_per_epoch * self.max_epochs, eta_min=self.final_lr)


    def update_lr(self):
        if self.schedular is not None:
            self.schedular.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_lr(self, param_group=0):
        return self.optimizer.param_groups[param_group]['lr']
    
    def get_weight_decay(self, param_group=0):
        return self.optimizer.param_groups[param_group]['weight_decay']



class MoSiCOptimizer:
    def __init__(self, model_parameters_dict, init_lr, peak_lr, warmup_steps, grad_norm_clipping, max_iter):
        self.model_parameters_dict = model_parameters_dict
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.grad_norm_clipping = grad_norm_clipping
        self.weight_decay_values = cosine_scheduler(0.04, 0.4, max_iter)
        self.max_iter = max_iter
        
        self.optimizer = None
        self.scheduler = None
        self.current_step = 0
    
    def setup_optimizer(self, optimizer_type='AdamW'):
        init_weight_decay = self.weight_decay_values[0]
        if optimizer_type == 'AdamW':
            self.optimizer = AdamW(self.model_parameters_dict, weight_decay=init_weight_decay)
        elif optimizer_type == 'SGD':
            self.optimizer = SGD(self.model_parameters_dict, weight_decay=init_weight_decay)
        else:
            raise ValueError("Unsupported optimizer type. Choose 'Adam' or 'SGD'.")
    
    def setup_scheduler(self):
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_iter, eta_min=0)
    
    def step(self):
        self.optimizer.step()
        if self.grad_norm_clipping:
            parameters = []
            for param_dict in self.model_parameters_dict:
                parameters += param_dict['params']
            torch.nn.utils.clip_grad_norm_(parameters, self.grad_norm_clipping)
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            self.warmup_lr()
        else:
            self.update_lr()
        self.adjust_weight_decay()
    
    def warmup_lr(self):
        if self.current_step < self.warmup_steps:
            warmup_factor = self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.init_lr +  warmup_factor * (self.peak_lr - self.init_lr)
    
    def adjust_weight_decay(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            if param_group["weight_decay"] != 0:
                param_group["weight_decay"] = self.weight_decay_values[self.current_step]
    
    def update_lr(self):
        if self.scheduler is not None:
            self.scheduler.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def get_weight_decay(self):
        return self.optimizer.param_groups[0]['weight_decay']