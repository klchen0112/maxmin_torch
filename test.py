import torch
import maxmin
device = torch.device("cuda")
a = torch.randint(300, size=(10,)).float().to(device)


minV = torch.Tensor([10]).float().to(device)
maxV = torch.Tensor([20]).float().to(device)

groud_truth = torch.clamp(a, min=minV, max=maxV)

own = maxmin.own_max_min(a, min=minV, max=maxV)

if torch.mean(torch.abs(own - groud_truth)).item() > 10**(-5):
    print(a)
    print(own)
    print(groud_truth)
    print(torch.mean(torch.abs(own - groud_truth)))
    print("reduce single error")
    exit()


a = torch.randint(300, size=(5, 5)).float().to(device)

minV = torch.randint(300, size=(5,)).float().to(device)
maxV = torch.randint(300, size=(5,)).float().to(device)

minV,maxV = torch.min(maxV,minV),torch.max(maxV,minV)

own = maxmin.own_max_min(a, min=minV, max=maxV)



print(a)
print(minV)
print(maxV)

print(own)

print("all pass")
