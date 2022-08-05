# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import tempfile
from typing import Optional

import torch
import torch.nn as nn
from torch import distributed as dist

from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from optimizers.bff_optimizer import BFF_AdamW


class Net(nn.Module):
    def __init__(self, param_dtype=torch.bfloat16):

        torch.manual_seed(2022)
        torch.cuda.manual_seed(0)
        super().__init__()

        self.linear1 = nn.Linear(8, 16, dtype=param_dtype)
        self.linear2 = nn.Linear(16, 8, dtype=param_dtype)
        self.out = nn.Linear(8, 4, dtype=param_dtype)

    def forward(self, x):
        output = self.linear1(x)
        output = self.linear2(output)
        output = self.out(output)

        return self.out(nn.functional.selu(output))


class TestBFFOptimizer(FSDPTest):
    def _init_net(self, sharding_strategy, net=None):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        net = net if net is not None else torch.nn.Linear(1, 5, bias=False)
        return FSDP(
            net,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
        )

    def _train_step(self, inpt, net, optim):
        optim.zero_grad()
        loss = net(inpt).sum()
        loss.backward()
        optim.step()

    def _check_grads_eq_rank(self, net, inpt):
        net.zero_grad()
        loss = net(inpt).sum()
        loss.backward()
        self.assertEqual(net.params[0].grad[0], self.rank)


instantiate_parametrized_tests(TestBFFOptimizer)
if __name__ == "__main__":
    run_tests()
