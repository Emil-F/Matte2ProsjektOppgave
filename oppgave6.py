import numpy as np
import matplotlib.pyplot as plt

alpha = 1.5e-7
T_ovn = 200.0
T_init = 15.0

Lx, Ly = 0.20, 0.10
Nx, Ny = 60, 30

x = np.linspace(0, Lx, Nx+2)
y = np.linspace(0, Ly, Ny+2)
dx = x[1]-x[0]
dy = y[1]-y[0]
X, Y = np.meshgrid(x, y)

dt = 0.9 * 0.5 / (alpha * (1/dx**2 + 1/dy**2))
rx = alpha*dt/dx**2
ry = alpha*dt/dy**2
T  = 60*60  

print(f"dx={dx*100:.2f} cm, dy={dy*100:.2f} cm, dt={dt:.1f} s, rx+ry={rx+ry:.3f}")

u = np.full((Ny+2, Nx+2), T_init)
u[0,:] = u[-1,:] = u[:,0] = u[:,-1] = T_ovn

plot_tider = [0, 2, 10, 30, 60]
fig, axes = plt.subplots(1, len(plot_tider), figsize=(18, 3))

def plott(u, t_min, ax):
    cf = ax.contourf(X*100, Y*100, u, levels=50, cmap='RdBu_r',
                     vmin=T_init, vmax=T_ovn)
    ax.set_title(f"t = {t_min} min")
    ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
    return cf

cf = plott(u, 0, axes[0])
plottet = {0: True, 2: False, 10: False, 30: False, 60: False}

t = 0.0
while t < T:
    uxx = u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]
    uyy = u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]
    u[1:-1,1:-1] += rx*uxx + ry*uyy
    u[0,:] = u[-1,:] = u[:,0] = u[:,-1] = T_ovn
    t += dt

    for tp in plot_tider:
        if not plottet[tp] and abs(t - tp*60) < dt/2:
            plott(u, tp, axes[plot_tider.index(tp)])
            plottet[tp] = True

plt.colorbar(cf, ax=axes[-1], label="Temperatur [°C]")
plt.suptitle("2D varmelikning – brødsteking (Forward Euler)", y=1.02)
plt.tight_layout()
plt.show()
