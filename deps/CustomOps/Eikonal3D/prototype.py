import numpy as np 
import matplotlib.pyplot as plt 
import tqdm

# |\nabla u| = f

# ((u - a1)^+)^2 + ((u - a2)^+)^2 + ((u - a3)^+)^2 = f^2 h^2

def calculate_unique_solution(a1, a2, a3, f, h):
    a1, a2, a3 = np.sort([a1, a2, a3])
    x = a1 + f * h
    if x <= a2:
        return x 
    else:
        B = -(a1+a2)
        C = (a1**2 + a2**2 - f**2 * h**2)/2
        x1 = (-B + np.sqrt(B**2 - 4*C))/2.0
        x2 = (-B - np.sqrt(B**2 - 4*C))/2.0
        if x1>a2:
            x = x1 
        else:
            x = x2 
        if x<=a3:
            return x 
        else:
            B = -2.0*(a1+a2+a3)/3.0
            C = (a1**2+a2**2+a3**2-f**2*h**2)/3.0
            x1 = (-B + np.sqrt(B**2 - 4*C))/2.0
            x2 = (-B - np.sqrt(B**2 - 4*C))/2.0
            if x1>a3:
                x = x1 
            else:
                x = x2
        return x 

def sweeping_over_I_J_K(u, I, J, K, f, h):
    # print("Sweeping start...")
    m = len(I)
    n = len(J)
    l = len(K)
    for i in I:
        for j in J:
            for k in K:
                if i==0:
                    uxmin = u[i+1,j,k]
                elif i==m-1:
                    uxmin = u[i-1,j,k]
                else:
                    uxmin = np.min([u[i-1,j,k], u[i+1,j,k]])

                if j==0:
                    uymin = u[i,j+1,k]
                elif j==n-1:
                    uymin = u[i,j-1,k]
                else:
                    uymin = np.min([u[i,j-1,k], u[i,j+1,k]])

                if k==0:
                    uzmin = u[i,j,k+1]
                elif k==l-1:
                    uzmin = u[i,j,k-1]
                else:
                    uzmin = np.min([u[i,j,k-1], u[i,j,k+1]])

                u_new = calculate_unique_solution(uxmin, uymin, uzmin, f[i,j,k], h)

                u[i,j,k] = np.min([u_new, u[i,j,k]])
    return u 


def sweeping(u, f, h):
    m, n, l = u.shape
    I = list(range(m)); iI = I[::-1]
    J = list(range(n)); iJ = J[::-1]
    K = list(range(l)); iK = K[::-1]

    # u = sweeping_over_I_J_K(u, I, J, K, f, h)
    # u = sweeping_over_I_J_K(u, I, iJ, K, f, h)
    # u = sweeping_over_I_J_K(u, I, iJ, iK, f, h)
    # u = sweeping_over_I_J_K(u, I, J, iK, f, h)




    # u = sweeping_over_I_J_K(u, I, J, K, f, h)
    # u = sweeping_over_I_J_K(u, iI, J, K, f, h)
    # u = sweeping_over_I_J_K(u, iI, J, iK, f, h)
    # u = sweeping_over_I_J_K(u, I, J, iK, f, h)

    u = sweeping_over_I_J_K(u, I, J, K, f, h)
    u = sweeping_over_I_J_K(u, iI, J, K, f, h)
    u = sweeping_over_I_J_K(u, iI, iJ, K, f, h)
    u = sweeping_over_I_J_K(u, I, iJ, K, f, h)

    u = sweeping_over_I_J_K(u, I, iJ, iK, f, h)
    u = sweeping_over_I_J_K(u, I, J, iK, f, h)
    u = sweeping_over_I_J_K(u, iI, J, iK, f, h)
    u = sweeping_over_I_J_K(u, iI, iJ, iK, f, h)




    return u 


def eikonal_solve(u, f, h):
    for i in range(50):
        u_old = np.copy(u)
        u = sweeping(u, f, h)

        err = np.max(np.abs(u - u_old))
        print(f"Iteration {i}, Error = {err}")
        if err < 1e-6:
            break 
    return u 


m = 21
n = 21
l = 21
f = np.ones((m, n, l))
# f[m//2:, n//2:, :] = 0.
h = 0.01

u = 1000*np.ones((m, n, l))
u[m//2, n//2, l//2] = 0.0

u = eikonal_solve(u, f, h)

plt.close("all")
plt.pcolormesh(u[m//2,:,:])
plt.colorbar()
plt.savefig("slice_x.png")

plt.close("all")
plt.pcolormesh(u[:,n//2,:])
plt.colorbar()
plt.savefig("slice_y.png")


plt.close("all")
plt.pcolormesh(u[:,:,l//2,])
plt.colorbar()
plt.savefig("slice_z.png")