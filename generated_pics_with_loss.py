import torch
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from scipy import sparse

import matplotlib.pyplot as plt

def build_laplace_2D_kron(N):
    D = sparse.diags([np.ones(N - 1), np.ones(N) * -2, np.ones(N - 1)],
                     [-1, 0, 1]).tocsr()
    I = sparse.eye(N).tocsr()
    return (sparse.kron(D, I) + sparse.kron(I, D)).toarray()

N = 62
A = build_laplace_2D_kron(N)*(N+1)**2

def laplace_zeroBC(f):
    rhs_matrix = f[1:-1,1:-1]
    numerical_solution = np.zeros((N+2,N+2))
    numerical_solution[1:-1,1:-1] = np.linalg.solve(A, rhs_matrix.reshape(-1)).reshape(N,N)

    return numerical_solution

a = torch.load("trained_models/UNet_poisson-1000_steps-250-sampling_steps-class_condn_False.pt")

# Create a meshgrid for the x and y coordinates
x = np.arange(0, 64)
y = np.arange(0, 64)
xx, yy = np.meshgrid(x, y)

for k in range(a.shape[0]):
    fig, ax = plt.subplots(1,3,figsize=(14, 4))

    data_point = torch.load("dataset/Poisson/seed_"+str(10001+k)+".pt")

    true_u = data_point[0,:,:].detach().numpy()
    true_f = data_point[1,:,:].detach().numpy()

    u = a[k,0].detach().numpy()
    # f = a[k,1].detach().numpy()

    ax[0].pcolormesh(xx, yy, true_u, cmap='viridis')
    ax[1].pcolormesh(xx, yy, u, cmap='viridis')
    ax[2].pcolormesh(xx, yy, true_f, cmap='viridis')
    # ax[3].pcolormesh(xx, yy, f, cmap='viridis')

    ax[0].set(title="True u")
    ax[1].set(title="DDIM u")
    ax[2].set(title="True f")
    # ax[3].set(title="DDIM f")

    # Add a colorbar to each subplot
    cbar1 = fig.colorbar(ax[0].imshow(true_u, cmap='viridis'), ax=ax[0])
    cbar2 = fig.colorbar(ax[1].imshow(u, cmap='viridis'), ax=ax[1])
    cbar3 = fig.colorbar(ax[2].imshow(true_f, cmap='viridis'), ax=ax[2])
    # cbar4 = fig.colorbar(ax[3].imshow(f, cmap='viridis'), ax=ax[3])

    plt.savefig("pde_figs/fig_"+str(k)+".png")
    plt.clf()