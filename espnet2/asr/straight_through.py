import torch
import logging

class OneHotArgmaxSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        indices = torch.argmax(logits, dim=2)
        one_hot = torch.zeros_like(logits)
        indices = indices.unsqueeze(2)
        one_hot.scatter_(2, indices, 1.0)
        return one_hot
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class DiscreteSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        return torch.argmax(logits, dim=2).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output