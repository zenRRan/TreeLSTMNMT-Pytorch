import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from driver.Config import *


class Optim(object):
    def __init__(self, config):
        self.last_ppl = None
        self.last_bleu = 0.0
        self.lr = config.learning_rate
        self.original_lr = config.learning_rate
        self.max_grad_norm = config.clip
        self.method = config.learning_algorithm
        self.lr_decay = config.decay
        self.start_decay_at = config.start_decay_at
        self.start_decay = False
        self._step = 0
        self.betas = [config.beta_1, config.beta_2]
        self.epsilon = config.epsilon

    def setRate(self, lr):
        self.lr = lr
        self.optimizer.param_groups[0]['lr'] = self.lr

    def set_parameters(self, params):
        self.params = [p for p in params if p.requires_grad]
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, betas=self.betas, eps=self.epsilon)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def step(self):
        "Compute gradients norm."
        self._step += 1

        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.step()

    def clearHistory(self):
        if self.method != 'adam':
            return

        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                else:
                    state['step'] = 0
                    state['exp_avg'].zero_()
                    state['exp_avg_sq'].zero_()


    def updateLearningRate(self, ppl, epoch):
        """
        Decay learning rate if val perf does not improve
        or we hit the start_decay_at limit.
        """

        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            if self.last_ppl is not None and ppl > self.last_ppl:
                self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_ppl = ppl
        self.optimizer.param_groups[0]['lr'] = self.lr
        
    def updateLearningRateByBleu(self, bleu):
        if bleu < self.last_bleu:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_bleu = bleu
        self.optimizer.param_groups[0]['lr'] = self.lr