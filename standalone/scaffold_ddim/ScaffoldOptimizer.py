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
            for p in group['params']:
                if p.grad is None:
                    continue

                # Retrieve control variates by parameter name
                param_name = next(n for n, param in server_controls.items() if param is p)

                # Check for shape alignment and apply control variate
                c, ci = server_controls[param_name], client_controls[param_name]
                if p.shape != c.shape or p.shape != ci.shape:
                    raise RuntimeError(f"Shape mismatch: parameter shape {p.shape}, server control shape {c.shape}, "
                                       f"client control shape {ci.shape}")

                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp * group['lr']

        return loss

