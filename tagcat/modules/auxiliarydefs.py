import torch

def repeat_tile(self, repeats, dim=0):
  shape = torch.ones(len(self.shape), dtype=int)
  shape[dim] = repeats
  return self.repeat(*shape)

torch.Tensor.repeat_tile = repeat_tile

def repeat_tile(a, repeats, dim=0):
  return a.repeat_tile(repeats, dim)

torch.repeat_tile = repeat_tile