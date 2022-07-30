# BFF_Optimizer: a pure Bfloat16 AdamW optimizer with optional Kahan summation and direct control over
# momentum, variance and auxiliary compensation buffer
# we use Kahan summarization to offset the Bfloat16 precision reduction, allowing full training in BFloat16.

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
        use_kahan_summation=True,
        # use_matching_params_dtype=False,
        momentum_dtype = torch.bfloat16,
        variance_dtype=torch.bfloat16,
        compensation_buffer_dtype = torch.bfloat16,
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
                use_kahan_summation = creates auxiliary buffer to ensure high precision model param updates
                # use_matching_params_dtype = should the optimizer use the same dtype as model params? True = regular AdamW
                momentum_dtype = dtype for momentum
                variance_dtype = dtype for uncentered variance
                compensation_buffer_dtype = dtype for Kahan summation buffer


        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_kahan_summation=use_kahan_summation,
            momentum_dtype = momentum_dtype,
            variance_dtype = variance_dtype,
            compensation_buffer_dtype = compensation_buffer_dtype
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
            # BFF specifics
            use_kahan_summation = group["use_kahan_summation"]

            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype=group["compensation_buffer_dtype"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("BFF does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:

                    #if use_kahan_summation:
                    #    assert (
                    #       p.dtype == torch.bfloat16
                    #    ), "BFF recommends BFloat16 datatype"

                    state["step"] = torch.tensor(0.0)

                    # todo - add match weights option...for now let user select
                    param_dtype = p.dtype

                    # EMA of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                        dtype=momentum_dtype,
                    )

                    # EMA of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                        dtype=variance_dtype,
                    )

                    # optional Kahan summation - accumulated error tracker
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, 
                            dtype=compensation_buffer_dtype
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
