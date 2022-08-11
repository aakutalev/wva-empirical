import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, shape=None, learning_rate=0.001, device=torch.device("cpu"), empirical_fisher=False):
        super(Model, self).__init__()

        self.lr = learning_rate
        self.shape = shape
        self.lmbda = 0.
        self.device = device
        self.empirical_fisher = empirical_fisher

        # create network layers
        self.network = nn.ModuleList()
        for ins, outs in zip(self.shape[:-1], self.shape[1:]):
            ins_ = np.abs(ins)
            outs_ = np.abs(outs)
            self.network.append(nn.Linear(ins_, outs_).to(self.device))

        # initialize optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.lr)

        self.params = [p for p in self.network.parameters() if p.requires_grad]

        # create list for parameter importances
        self.importances = [torch.zeros_like(p, device=self.device) for p in self.params]

        # temporary accumulators for importances
        self.accumulators = [torch.zeros_like(p, device=self.device) for p in self.params]

    def reset(self):
        # reinitialize network
        for layer in self.network:
            layer.reset_parameters()

        # clear importance
        for imp in self.importances:
            imp.fill_(0.)

        # reinitialize optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.lr)

    def forward(self, inputs):
        last_layer_idx = len(self.network) - 1
        for i, layer in enumerate(self.network):
            z = layer(inputs)
            if i < last_layer_idx:
                inputs = F.leaky_relu(z)
        return z

    def prepare_for_step(self):
        self.copies = [p.clone().detach_() for p in self.params]

    def correct_step(self):
        for p, prev, att in zip(self.params, self.copies, self.attenuators):
            p = p.data
            p.add_((prev - p) * att)
        del self.copies

    def step(self, inputs, labels):
        z = self.forward(inputs)
        loss = F.cross_entropy(z, labels)

        #if self.lmbda != 0. and self.star_params is not None:
        #    for p, reg, p_star in zip(list(self.network.parameters()), self.regularizers, self.star_params):
        #        loss += torch.sum(reg * torch.square(p - p_star))

        self.optimizer.zero_grad()
        loss.backward()
        self.prepare_for_step()
        self.optimizer.step()
        self.correct_step()

    def open_lesson(self, lmbda=0.0):
        """
        :param lmbda:         ewc regularization power
        """
        self.lmbda = lmbda
        self.attenuators = [1.0 - 1.0 / (imp * lmbda + 1.0) for imp in self.importances]

    def close_lesson(self, inputs=None, labels=None):
        """
        Закрытие урока обучения сети на отдельном датасете. Расчет и накопление важностей весов.
        :param closing_set: датасет, на котором будут рассчитаны важности весов после обучения
        :return:
        """
        if inputs is None:
            return

        self.eval()

        s_num = len(inputs)

        for accum in self.accumulators:
            accum.fill_(0.)

        for i in range(s_num):
            _input = torch.tensor(inputs[i: i + 1], device=self.device)

            logits = self.forward(_input)
            log_probs = F.log_softmax(logits, dim=1)

            if self.empirical_fisher:
                label = labels[i]
            else:
                probs = F.softmax(logits[0])
                label = torch.multinomial(probs, 1, True)[0].item()

            self.optimizer.zero_grad()
            log_prob = log_probs[0, label]
            log_prob.backward()

            for accum, v in zip(self.accumulators, self.params):
                accum.add_(v.grad.square())

        for accum, imp in zip(self.accumulators, self.importances):
            imp.add_(accum / s_num)
