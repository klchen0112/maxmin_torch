import torch
import maxmin

a = torch.randint(300,shape=(100))


minV = 10
maxV = 20

groud_truth = torch.clamp(a,min=minV,max=maxV)

own  = maxmin.own_max_min(a,min=minV,max=maxV)

if torch.reduce_mean(torch.abs(own - groud_truth)).item() > 10**(-5):
    print("reduce single error")
    exit()


a = torch.randint(300,shape=(100,100))

minV = torch.randint(300,shape=(100,))
maxV = torch.randint(300,shape=(100,))

groud_truth = torch.clamp(a,min=minV,max=maxV)

own  = maxmin.own_max_min(a,min=minV,max=maxV)


if torch.reduce_mean(torch.abs(own - groud_truth).flatten()).item() > 10**(-5):
    print("reduce single error")
    exit()
