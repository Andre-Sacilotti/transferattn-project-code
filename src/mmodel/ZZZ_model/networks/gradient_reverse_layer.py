from torch import nn
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, back_coeff, lambda_=None):
        ctx.back_coeff = back_coeff
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        back_coeff = ctx.back_coeff
        if ctx.lambda_ is not None:
            # print(" USANDO P", ctx.lambda_)
            lambda_ = ctx.lambda_
            reverse_with_coeff = -grad_output * lambda_  # *back_coeff
            return reverse_with_coeff, None, None
        reverse_with_coeff = -grad_output * back_coeff
        return reverse_with_coeff, None, None


class GradReverseLayer(nn.Module):
    def __init__(self, coeff_fn=lambda: 1):
        super().__init__()
        self.coeff_fn = coeff_fn

    def forward(self, x, lambda_=None):
        x = GradReverse.apply(x, self.coeff_fn(), lambda_)
        return x
