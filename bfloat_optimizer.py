# BFF_Optimizer: a pure Bfloat16 optimizer - basic idea is we use Kahan summarization to offset the Bfloat16 precision reduction, allowing full training in BFloat16.

# paper credit - "Revisiting Bfloat16 training" - https://arxiv.org/abs/2010.06192
# original inspiration - https://github.com/arogozhnikov/adamw_bfloat16
# Kahan summation - https://en.wikipedia.org/wiki/Kahan_summation_algorithm

import torch
from torch.optim.optimizer import Optimizer


class BFF_Optimizer(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        enforce_bfloat_states=False,
        kahan_summation=False,
        stochastic_rounding=False,
    ):

        """
        Args:
                params (iterable): iterable of parameters to optimize or dicts defining
                    parameter groups
                lr (float, optional): learning rate (default: 1e-3)
                betas (Tuple[float, float], optional): coefficients used for computing
                    running averages of gradient and its square (default: (0.9, 0.999))
                eps (float, optional): term added to the denominator to improve
                    numerical stability (default: 1e-8)
                weight_decay (float, optional): weight decay coefficient (default: 1e-2)

                # BFF specific
                enforce_bfloat_states = whether states for variance and momentum are forced to bfloat16.
                If false, the datatype will mirror the weights in use.

        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            kahan_summation=kahan_summation,
            enforce_bfloat16_states=enforce_bfloat_states,
            stochastic_rounding=stochastic_rounding,
        )

        super().__init__(params, defaults)
        print(f"BFF Optimizer initialized with {defaults}")

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            kahan_summation = group["kahan_summation"]
            enforce_bf16_states = group["enforce_bfloat16_states"]
            stochastic_rounding = group["stochastic_rounding"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("BFF does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:

                    if kahan_summation or stochastic_rounding:
                        assert (
                            p.dtype == torch.bfloat16
                        ), "BFF requires BFloat16 datatype"

                    state["step"] = torch.tensor(0.0)

                    # handle what state dtype should be...if enforced, we set to bf16 else we match the weights
                    state_dtype = p.dtype

                    if enforce_bf16_states:
                        state_dtype = torch.bfloat16
                        # print(f"BFF state dtype set to torch.bfloat16")

                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                        dtype=state_dtype,
                    )

                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                        dtype=state_dtype,
                    )

                    # Kahan summation - accumulated error tracker
                    # enforce bfloat16 no matter what
                    if kahan_summation:
                        state["compensation"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, dtype=torch.bfloat16
                        )

                # main processing

                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                grad = p.grad

                # Decay the first and second moment running average coefficient
                """exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                """
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if kahan_summation:
                    compensation = state["compensation"]

                # weight decay, AdamW style - todo - this differs from torch impl
                if weight_decay:
                    p.data.mul_(1 - lr * weight_decay)

                denom_correction = (1 - beta2**step) ** 0.5

                # lr update to compensation
                if kahan_summation:
                    compensation.addcdiv_(
                        exp_avg,
                        exp_avg_sq.sqrt().add_(eps, alpha=1),
                        value=-lr * denom_correction,
                    )

                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    buffer = p.clone()
                    p.add_(compensation)
                    compensation.add_(buffer.sub_(p))

                else:
                    # standard update
                    p.data.addcdiv_(
                        exp_avg,
                        exp_avg_sq.sqrt().add_(eps, alpha=1),
                        value=-lr * denom_correction,
                    )
