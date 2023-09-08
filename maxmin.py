import math
from torch import nn
from torch.autograd import Function
import torch

import maxmin_cuda


def own_max_min(input, min, max):
    return maxmin_cuda.own_max_min(input, min, max)
