from torch.autograd.function import Function


class Softsign(Function):

    def forward(self, input):
        self.buffer = input.clone().abs_().add_(1)
        self.buffer_squared = False
        return input.clone().div_(self.buffer)

    def backward(self, grad_output):
        if not self.buffer_squared:
            self.buffer.mul_(self.buffer)
            self.buffer_squared = True
        return grad_output.clone().div_(self.buffer)
