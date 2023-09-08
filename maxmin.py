import math
from torch import nn
from torch.autograd import Function
import torch

import maxmin_cuda



def own_max_min(input,maxT,minT):
    return maxmin_cuda.forward(input, maxT,minT)


