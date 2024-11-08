# -*- coding:utf-8 -*-
"""
@Time: 2022/03/02 13:34
@Author: KI
@File: ScaffoldOptimizer.py
@Motto: Hungry And Humble
"""
from torch.optim import Optimizer


class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Iterate over parameter groups and parameters
        for group in self.param_groups:
            for p, param_name in zip(group['params'], group['name']):
                if p.grad is None:
                    continue

                # Retrieve control variates by parameter name
                c = server_controls[param_name]
                ci = client_controls[param_name]

                # Check for shape alignment and apply control variate
                if p.shape != c.shape or p.shape != ci.shape:
                    raise RuntimeError(f"Shape mismatch: parameter shape {p.shape}, server control shape {c.shape}, "
                                       f"client control shape {ci.shape}")

                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp * group['lr']

        return loss
