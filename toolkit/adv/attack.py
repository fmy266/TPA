#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author：fmy
import torch
import torchvision
from .utils import clip_based_on_inf_norm_, clip_based_on_2_norm_
from functools import partial


class BaseAttack:
    def __init__(self, epsilon, step_size=0.05, iter_num=40, device=torch.device("cuda:0")):
        self.epsilon = epsilon
        self.step_size = step_size
        self.iter_num = iter_num
        self.device = device

    def step(self, data, distortion, label, substitute_model, loss_func, target: bool = False):
        output = substitute_model(data + distortion) #  + torch.randn_like(data, device=self.device)/100.
        if target:
            loss = loss_func(output, label)
        else:
            loss = -loss_func(output, label) # 记着改回来
        loss.backward()
        return distortion.grad

    def produce_adv(self, data, label, substitute_model, loss_func, target: bool = False):
        distortion = self.distortion_generation(data)
        optimizer = self.get_optim(distortion)
        for _ in range(self.iter_num):
            optimizer.zero_grad()
            distortion.grad = self.step(data, distortion, label, substitute_model, loss_func, target)
            self.grad_transform(distortion)
            optimizer.step()
            self.clip(distortion)
        return distortion

    @torch.no_grad()
    def get_optim(self, data):
        return torch.optim.SGD([data, ], lr=self.step_size)

    @torch.no_grad()
    def distortion_generation(self, data):
        return torch.rand_like(data, device=self.device).div_(200.).requires_grad_(True)
        # return torch.zeros_like(data, device=self.device).requires_grad_(True)

    @torch.no_grad()
    def grad_transform(self, distortion):
        distortion.grad.data.sign_()

    @torch.no_grad()
    def clip(self, distortion):
        # clip_based_on_2_norm_(distortion, self.epsilon, self.device)
        clip_based_on_inf_norm_(distortion, self.epsilon, self.device)



class TAEFEP(BaseAttack):
    def __init__(self, epsilon, step_size=0.031, iter_num=10, device=torch.device("cuda:0"), alpha=1.,
                 noise_magn=5., forward_step_size=1., copies=20):

        super().__init__(epsilon=epsilon, step_size=step_size, iter_num=iter_num, device=device)
        self.alpha = alpha # 0 ~ 1
        self.noise_magn = noise_magn # 1, 2, 4, 8, 12, 16, 20
        self.forward_step_size = forward_step_size # 0.01, 0.03, 0.06, 0.09, 0.12, 0.15
        self.copies = copies # 5, 10, 15, 20

    def step(self, data, distortion, label, substitute_model, loss_func, target: bool = False):

        # compute gradients at current position
        output = substitute_model(data + distortion)
        if target:
            loss = loss_func(output, label)
        else:
            loss = -loss_func(output, label)
        loss.backward()
        cur_gradient = distortion.grad.detach().clone()

        forward_gradient = torch.zeros_like(cur_gradient) # 带有噪声向低损失区域一步，随机噪声并不可信；判断周围区域是否是否存在高置信度的样本
        for _ in range(self.copies):
            distortion.grad.zero_()

            random_noise = (torch.rand_like(distortion)-0.5)*2. * 0.031 * self.noise_magn # 0~0.1 approxi [0, 5/255] [-1,1]
            output = substitute_model(data + distortion + random_noise) # 扰动界限很重要

            if target:
                loss = loss_func(output, label)
            else:
                loss = -loss_func(output, label)
            loss.backward()

            temp_gradient = distortion.grad.detach().clone()
            distortion.grad.zero_()
            output = substitute_model(data + distortion + random_noise + self.forward_step_size * temp_gradient.sign())
            if target:
                loss = loss_func(output, label)
            else:
                loss = -loss_func(output, label)
            loss.backward()

            forward_gradient = forward_gradient + distortion.grad.detach().clone() # / self.forward_step_size

        forward_gradient = forward_gradient / self.copies

        return (1 - self.alpha) * cur_gradient + self.alpha * forward_gradient

    def get_optim(self, data):
        return torch.optim.SGD([data, ], lr=self.step_size, momentum=0.9)

