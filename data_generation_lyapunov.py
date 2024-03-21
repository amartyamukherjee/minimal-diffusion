import numpy as np
import torch
import torch.nn.functional as F
from torch import tanh
import sympy as sp
import lyznet

x = y = np.linspace(-1,1,64)
xx,yy = np.meshgrid(x,y)
xx_t,yy_t = torch.Tensor(xx),torch.Tensor(yy)
coords = np.stack((xx,yy)).reshape(2,-1)

def hurwitz_data_gen(m=20):
    # Define symbols
    x1, x2 = sp.symbols('x1 x2')
    x = sp.Matrix([x1, x2])
    symbolic_vars = [x1, x2]

    A = generate_hurwitz_matrix(2)
    f_nominal = A * x
    W_f = np.random.randn(m, 2)   
    beta_f = np.random.randn(2, m) * 0.2
    perturbation = lyznet.utils.generate_NN_perturbation_dReal_to_Sympy(
        W_f, beta_f, symbolic_vars)
    f_total = f_nominal + perturbation

    domain = [[-1, 1]]*2
    sys_name = "random_poly_2d.py"
    system = lyznet.DynamicalSystem(f_total, domain, sys_name)

    if system.P is not None:
        W_V, b_V, beta_V, model_path, max_test_loss = lyznet.numpy_elm_learner(
            system, num_hidden_units=800, num_colloc_pts=9000,
            lambda_reg=0.0, test=True, return_test_loss=True
            )

        if max_test_loss < 1e-6:

            c1_P = lyznet.local_stability_verifier(system)
            c2_P = lyznet.quadratic_reach_verifier(system, c1_P)

            successful_samples += 1
            model_path_with_number = f"{model_path}_system_{successful_samples}"
    
    return A,W_f,beta_f,W_V,b_V,beta_V


# Function to check if a matrix is Hurwitz
def is_hurwitz(matrix):
    eigenvalues = matrix.eigenvals()
    # Check if all real parts of eigenvalues are negative
    return all(re.evalf().as_real_imag()[0] < 0 for re in eigenvalues)


# Function to generate a random Hurwitz matrix of size 2x2
def generate_hurwitz_matrix(size):
    while True:
        A = np.random.randn(size, size)
        A_sym = sp.Matrix(A)
        if is_hurwitz(A_sym):
            return A_sym



def second_order_lyap_fn(i):
    # x1' = x2
    # x2' = -a x1 - tanh(b x1) - c x2 - tanh(d x2)
    # V = (a/2 x1^2 + ln(cosh(b x1))/b + x2^2 / 2) / c
    # V' < - x2^2
    # Constraints: a > 0, b > 0, c > 0, d > 0

    a = torch.rand(())*5
    b = torch.rand(())*5
    c = torch.rand(())*5
    d = torch.rand(())*5
    
    f1 = yy_t
    f2 = -a*xx_t - 20*tanh(b*xx_t) - c*yy_t - 20*tanh(d*yy_t)
    V = a/2 * xx_t**2 + torch.log(torch.cosh(b*xx_t))/b + yy_t**2/2

    m = f2.abs().max()
    if m > 1:
        f1 = f1 / m
        f2 = f2 / m

    V = V / V.abs().max()

    img = torch.stack((f1,f2,V))
    torch.save(img, "dataset/Lyapunov/seed_"+str(i)+".pt")

for i in range(1000):
    A,W_f,beta_f,W_V,b_V,beta_V = hurwitz_data_gen()

    f = A @ coords + beta_f @ np.tanh(W_f @ coords)
    f = f.reshape(2,64,64)
    f = np.float32(f)
    f = f / np.abs(f).max()

    V = beta_V.T @ np.tanh(W_V @ coords + b_V)
    V = V.reshape(1,64,64)
    V = np.float32(V)
    V = V / np.abs(V).max()

    img = np.concatenate((f,V), dtype=np.float32)
    img_torch = torch.Tensor(img)

    torch.save(img_torch, "dataset/Lyapunov/seed_"+str(2*i)+".pt")
    second_order_lyap_fn(2*i+1)
