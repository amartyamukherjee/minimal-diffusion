import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m = 64
x = torch.linspace(0.0, 1.0, m, device=device)
y = torch.linspace(0.0, 1.0, m, device=device)

x, y = torch.meshgrid(x, y)
pos = torch.stack((x, y), axis=2)
pos.requires_grad = True

def compute_laplacian(func, pos):
    u = func(pos)
    grad_x = torch.autograd.grad(outputs=u, inputs=pos, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    grad_xx = torch.autograd.grad(outputs=grad_x[...,0], inputs=pos, grad_outputs=torch.ones((m,m)), create_graph=True)[0]
    grad_yy = torch.autograd.grad(outputs=grad_x[...,1], inputs=pos, grad_outputs=torch.ones((m,m)), create_graph=True)[0]
    return grad_xx[...,0]+grad_yy[...,1]

############# DNN #############

def generate_sample(seed):
    torch.manual_seed(seed)
    print("Generating sample for seed: " + str(seed))

    dnn = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    ).to(device)

    def sol(pos):
        return dnn(pos)[..., 0] * pos[..., 0] * (1 - pos[..., 0]) * pos[..., 1] * (1 - pos[..., 1])

    u = sol(pos)

    f = compute_laplacian(sol, pos)
    combined = torch.stack((u, f), axis=0)
    combined = combined / combined.abs().max()

    print("Saved sample for seed: " + str(seed))

    return combined

for i in range(10001, 10001+64):
    samples = generate_sample(i)
    torch.save(samples.detach(), "dataset/Poisson/seed_"+str(i)+".pt")

# ############# Type 0 #############

# # Values for n and k to vary
# n_vals = range(1,30+1)
# k_vals = range(1,40+1)

# for n in n_vals:
#     for k in k_vals:
#         # Analytical solution
#         u = torch.sin(n * torch.pi * x) * torch.sin(k * torch.pi * y)
#         f = -torch.pi**2 * (n**2 + k**2) * torch.sin(n*torch.pi*x) * torch.sin(k*torch.pi*y)

#         combined = torch.stack((u,f), axis = 0)
#         combined = combined / combined.abs().max() * (2 * torch.randint(0,2,()) - 1)

#         torch.save(combined, "dataset/Poisson/type0_n_"+str(n)+"_k_"+str(k)+".pt")

# ############# Type 1 #############

# # Values for n and k and j to vary
# n_vals = range(1,20+1)
# k_vals = range(1,20+1)
# j_vals = range(1,20+1)

# for n in n_vals:
#     for k in k_vals:
#         for j in j_vals:
#             def fn(pos):
#                 return torch.sin(n*torch.pi*pos[...,0])*torch.sin(k*torch.pi*pos[...,1])*torch.sin(j*torch.pi*pos[...,0])
#             u = fn(pos)
#             f = compute_laplacian(fn, pos)

#             combined = torch.stack((u,f), axis = 0)
#             combined = combined / combined.abs().max() * (2 * torch.randint(0,2,()) - 1)

#             torch.save(combined, "dataset/Poisson/type1_n_"+str(n)+"_k_"+str(k)+"_j_"+str(j)+".pt")

# ############# Type 2 #############

# # Values for n and k to vary
# n_vals = range(1,30+1)
# k_vals = range(1,30+1)

# for n in n_vals:
#     for k in k_vals:
#         def fn(pos):
#             return torch.sin(n*torch.pi*pos[...,0])*torch.sin(k*torch.pi*pos[...,1])*torch.cos(n*torch.pi*pos[...,0])
#         u = fn(pos)
#         f = compute_laplacian(fn, pos)

#         combined = torch.stack((u,f), axis = 0)
#         combined = combined / combined.abs().max() * (2 * torch.randint(0,2,()) - 1)

#         torch.save(combined, "dataset/Poisson/type2_n_"+str(n)+"_k_"+str(k)+".pt")

# ############# Type 3 #############

# # Values for n and k to vary
# n_vals = range(1,30+1)
# k_vals = range(1,30+1)

# for n in n_vals:
#     for k in k_vals:
#         def fn(pos):
#             return torch.sin(n*torch.pi*pos[...,0])*torch.sin(k*torch.pi*pos[...,1])*torch.cos(n*torch.pi*pos[...,1])
#         u = fn(pos)
#         f = compute_laplacian(fn, pos)

#         combined = torch.stack((u,f), axis = 0)
#         combined = combined / combined.abs().max() * (2 * torch.randint(0,2,()) - 1)

#         torch.save(combined, "dataset/Poisson/type3_n_"+str(n)+"_k_"+str(k)+".pt")

# ############# Type 4 #############

# # Values for n and k and j to vary
# n_vals = range(1,20+1)
# k_vals = range(1,20+1)
# j_vals = range(1,20+1)

# for n in n_vals:
#     for k in k_vals:
#         for j in j_vals:
#             def fn(pos):
#                 return torch.sin(n*torch.pi*pos[...,0])*torch.sin(k*torch.pi*pos[...,1])*torch.cos(j*torch.pi*pos[...,0])

#             u = fn(pos)
#             f = compute_laplacian(fn, pos)

#             combined = torch.stack((u,f), axis = 0)
#             combined = combined / combined.abs().max() * (2 * torch.randint(0,2,()) - 1)

#             torch.save(combined, "dataset/Poisson/type4_n_"+str(n)+"_k_"+str(k)+"_j_"+str(j)+".pt")

# ############# Type 5 #############

# # Values for n and k and j to vary
# n_vals = range(1,20+1)
# k_vals = range(1,20+1)
# j_vals = range(1,20+1)

# for n in n_vals:
#     for k in k_vals:
#         for j in j_vals:
#             def fn(pos):
#                 return torch.sin(n*torch.pi*pos[...,0])*torch.sin(k*torch.pi*pos[...,1])*torch.cos(j*torch.pi*pos[...,1])

#             u = fn(pos)
#             f = compute_laplacian(fn, pos)

#             combined = torch.stack((u,f), axis = 0)
#             combined = combined / combined.abs().max() * (2 * torch.randint(0,2,()) - 1)

#             torch.save(combined, "dataset/Poisson/type5_n_"+str(n)+"_k_"+str(k)+"_j_"+str(j)+".pt")

# ############# Type 6 #############

# # Values for n and k and j to vary
# n_vals = range(1,5+1)
# k_vals = range(1,5+1)
# i_vals = range(1,5+1)
# j_vals = range(1,5+1)

# for n in n_vals:
#     for k in k_vals:
#         for i in i_vals:
#             for j in j_vals:
#                 def fn(pos):
#                     x1 = pos[...,0]
#                     y1 = pos[...,1]
#                     return torch.pow(x1,n)*torch.pow((1-x1),k)*torch.pow(y1,i)*torch.pow((1-y1),j)*torch.exp(x1-y1)

#                 u = fn(pos)
#                 f = compute_laplacian(fn, pos)

#                 combined = torch.stack((u,f), axis = 0)
#                 combined = combined / combined.abs().max() * (2 * torch.randint(0,2,()) - 1)

#                 torch.save(combined, "dataset/Poisson/type6_n_"+str(n)+"_k_"+str(k)+"_i_"+str(i)+"_j_"+str(j)+".pt")

# ############# Type 7 #############

# # Values for n and k and j to vary
# n_vals = range(1,5+1)
# k_vals = range(1,5+1)
# i_vals = range(1,5+1)
# j_vals = range(1,5+1)

# for n in n_vals:
#     for k in k_vals:
#         for i in i_vals:
#             for j in j_vals:
#                 def fn(pos):
#                     x1 = pos[...,0]
#                     y1 = pos[...,1]
#                     return torch.pow(x1,n)*torch.pow((1-x1),k)*torch.pow(y1,i)*torch.pow((1-y1),j)*torch.exp(y1-x1)

#                 u = fn(pos)
#                 f = compute_laplacian(fn, pos)

#                 combined = torch.stack((u,f), axis = 0)
#                 combined = combined / combined.abs().max() * (2 * torch.randint(0,2,()) - 1)

#                 torch.save(combined, "dataset/Poisson/type7_n_"+str(n)+"_k_"+str(k)+"_i_"+str(i)+"_j_"+str(j)+".pt")
