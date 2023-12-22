import os
import torch
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator

def build_laplace_2D_kron(N):
    D = sparse.diags([np.ones(N - 1), np.ones(N) * -2, np.ones(N - 1)],
                     [-1, 0, 1]).tocsr()
    I = sparse.eye(N).tocsr()
    return (sparse.kron(D, I) + sparse.kron(I, D)).toarray()

N = 62
A = build_laplace_2D_kron(N)*(N+1)**2
u,s,v = np.linalg.svd(A)
# print(u)
# print(s)
# print(v)

pts = np.stack([i for i in np.meshgrid(np.linspace(0,1,64),np.linspace(0,1,64))], axis=2)

def laplace_zeroBC(f):
    rhs_matrix = f[1:-1,1:-1]
    numerical_solution = np.zeros((N+2,N+2))
    numerical_solution[1:-1,1:-1] = np.linalg.solve(A, rhs_matrix.reshape(-1)).reshape(N,N)
    return numerical_solution

# a = torch.load("trained_models/UNet_poisson-1000_steps-100-sampling_steps-class_condn_False.pt")
# a = torch.load("trained_models/UNet_poisson-1000_steps-250-sampling_steps-class_condn_False_f1.pt")
# a = torch.load("trained_models/UNet_poisson-1000_steps-2-sampling_steps-class_condn_False.pt")
a = torch.load("trained_models/UNet_poisson-1000_steps-2-sampling_steps-class_condn_False.pt")

# Create a meshgrid for the x and y coordinates
x = np.arange(0, 64)
y = np.arange(0, 64)
xx, yy = np.meshgrid(x, y)

def laplacian_spectral_bc(u,f,sigma=100000):
    # Compute the 2D Fourier transform of u
    u_hat = np.fft.fft2(u[:,:])
    f_hat = np.fft.fft2(f)
    # u_hat = u_hat - np.real(u_hat)

    # Define the wavenumbers
    kx = 2 * np.pi * np.fft.fftfreq(64, d=1/63)
    ky = 2 * np.pi * np.fft.fftfreq(64, d=1/63)

    kx,ky = np.meshgrid(kx, ky)

    k_indices = (kx**2 + ky**2 > sigma)
    kx[k_indices] = 0
    ky[k_indices] = 0

    # Enforce boundary conditions in Fourier space (set corresponding modes to zero)
    # u_hat[:, 0] = u_hat[:, -1] = u_hat[0, :] = u_hat[-1, :] = 0

    # Compute Laplacian in Fourier space
    laplacian_u_hat = u_hat #- np.real(u_hat)
    laplacian_u_hat = -(kx**2 + ky**2) * laplacian_u_hat# + np.real(u_hat)

    # print(np.mean(np.abs(np.imag(laplacian_u_hat)-np.imag(f_hat)), dtype=np.float128))

    # Inverse transform to get the Laplacian in physical space

    # laplacian_u[1:-1,1:-1] = np.real(np.fft.ifft2(laplacian_u_hat))
    laplacian_u = np.real(np.fft.ifft2(laplacian_u_hat))

    return np.clip(laplacian_u,-1,1)

def laplacian_fd_bc(u):
    fd = (A @ u[1:-1,1:-1].reshape(-1)).reshape(62,62)

    rgi = RegularGridInterpolator((np.linspace(0,1,62),np.linspace(0,1,62)),fd)

    return rgi(pts).T

# a = None
mse_diff = 0
mse_fd = 0

for k in range(a.shape[0]):
# for k in range(1):
    fig, ax = plt.subplots(1,2,figsize=(10, 4))

    data_point = torch.load("dataset/Poisson/seed_"+str(10001+k)+".pt")

    true_u = data_point[0,:,:].detach().numpy()
    true_f = data_point[1,:,:].detach().numpy()

    u = a[k,0].detach().numpy()
    f = a[k,1].detach().numpy()

    # print(u)
    # print(f)

    # f_fd = laplacian_fd_bc(true_u)
    u_fd = laplace_zeroBC(true_f)

    # f = poisson(true_f,nptx,npty,dx,dy)

    # f = laplacian_spectral_bc(true_u,true_f)

    # mse_diff += np.mean(np.abs(f-true_f))
    # mse_fd += np.mean(np.abs(f_fd-true_f))
    # mse_diff += np.mean(np.abs(f-true_f), dtype=np.float128)
    # mse_fd += np.mean(np.abs(f_fd-true_f), dtype=np.float128)
    mse_fd += np.mean(np.abs(u_fd-u), dtype=np.float128)

    # print(np.mean(np.square(f-true_f)) - np.mean(np.square(f_fd-true_f)))
    # print(np.mean(np.abs(f-true_f)) - np.mean(np.abs(f_fd-true_f)))

    f = true_f

    ax[0].pcolormesh(xx, yy, true_u, cmap='viridis')
    ax[1].pcolormesh(xx, yy, f, cmap='viridis')
    # ax[2].pcolormesh(xx, yy, true_f, cmap='viridis')
    # ax[3].pcolormesh(xx, yy, f, cmap='viridis')

    ax[0].set(title="u")
    ax[1].set(title="f")
    # ax[0].set(title="True u")
    # ax[1].set(title="DDIM f")
    # ax[2].set(title="True f")
    # ax[3].set(title="DDIM f")

    # Add a colorbar to each subplot
    cbar1 = fig.colorbar(ax[0].imshow(true_u, cmap='viridis'), ax=ax[0])
    cbar2 = fig.colorbar(ax[1].imshow(f, cmap='viridis'), ax=ax[1])
    # cbar3 = fig.colorbar(ax[2].imshow(true_f, cmap='viridis'), ax=ax[2])
    # cbar4 = fig.colorbar(ax[3].imshow(f, cmap='viridis'), ax=ax[3])

    os.makedirs("pde_figs_generated", exist_ok=True)
    plt.savefig("pde_figs_generated/fig_"+str(k)+".png")
    plt.clf()

mse_diff = mse_diff/64
mse_fd = mse_fd/64
print(mse_diff, mse_fd, mse_diff-mse_fd)