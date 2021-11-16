import torch
import torch.nn as nn

class grl_func(torch.autograd.Function):
    def __init__(self):
        super(grl_func, self).__init__()
    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None

class GRL(nn.Module):
    def __init__(self, gamma = 10):
        super(GRL, self).__init__()
        self.gamma = gamma

    def forward(self, x,p):

        lambda_ = (2 / (1+ torch.exp(-self.gamma*p[0])))-1

        return grl_func.apply(x, lambda_)
