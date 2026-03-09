import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from IPython.display import HTML


# PARAMETRE (identiske med 7a)

alpha_brod = 1.5e-7
alpha_luft = 2.2e-5
T_ovn      = 200.0

Lx_brod = 0.20;  Ly_brod = 0.10
Lx_luft = Lx_brod / 2;  Ly_luft = Ly_brod / 2

x_min = 0.0;  x_max = Lx_brod + 2*Lx_luft
y_min = 0.0;  y_max = Ly_brod + 2*Ly_luft

x_brod_min = Lx_luft;  x_brod_max = Lx_luft + Lx_brod
y_brod_min = Ly_luft;  y_brod_max = Ly_luft + Ly_brod

Nx = 121;  Ny = 61
x  = np.linspace(x_min, x_max, Nx)
y  = np.linspace(y_min, y_max, Ny)
dx = x[1] - x[0];  dy = y[1] - y[0]
X2d, Y2d = np.meshgrid(x, y)

i_brod = ((X2d >= x_brod_min) & (X2d <= x_brod_max) &
          (Y2d >= y_brod_min) & (Y2d <= y_brod_max))

alpha  = np.where(i_brod, alpha_brod, alpha_luft)
dt_max = 0.5 / (alpha_luft * (1/dx**2 + 1/dy**2))
dt     = 0.9 * dt_max


# SIMULERING

U = np.where(i_brod, 15.0, 200.0).astype(float)
U[0, :] = T_ovn;  U[-1, :] = T_ovn
U[:, 0] = T_ovn;  U[:, -1] = T_ovn

t_slutt    = 150 * 60
n_frames   = 120
n_steps    = int(t_slutt / dt)
lagre_hver = max(1, n_steps // n_frames)

frames = [];  tider = [];  t = 0.0

for n in range(n_steps):
    Un  = U.copy()
    uxx = (U[1:-1, 2:] - 2*U[1:-1, 1:-1] + U[1:-1, :-2]) / dx**2
    uyy = (U[2:, 1:-1] - 2*U[1:-1, 1:-1] + U[:-2, 1:-1]) / dy**2
    Un[1:-1, 1:-1] = U[1:-1, 1:-1] + dt * alpha[1:-1, 1:-1] * (uxx + uyy)
    Un[0, :] = T_ovn;  Un[-1, :] = T_ovn
    Un[:, 0] = T_ovn;  Un[:, -1] = T_ovn
    U = Un;  t += dt
    if n % lagre_hver == 0:
        frames.append(U.copy());  tider.append(t)

print(f"Simulering ferdig – {len(frames)} frames, siste t = {tider[-1]/60:.1f} min")


# ANIMASJON

norm = Normalize(vmin=15, vmax=200)
fig, ax = plt.subplots(figsize=(8, 5))

cf = ax.contourf(X2d*100, Y2d*100, frames[0], levels=50, cmap='RdBu_r', norm=norm)
plt.colorbar(cf, ax=ax, label='Temperatur [°C]')
ax.set_xlabel('x [cm]');  ax.set_ylabel('y [cm]')

def oppdater(k):
    ax.cla()
    ax.contourf(X2d*100, Y2d*100, frames[k], levels=50, cmap='RdBu_r', norm=norm)
    ax.add_patch(mpatches.Rectangle(
        (x_brod_min*100, y_brod_min*100), Lx_brod*100, Ly_brod*100,
        edgecolor='black', facecolor='none', lw=2))
    ax.set_title(f"t = {tider[k]/60:.1f} min")
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    return []

anim = animation.FuncAnimation(fig, oppdater, frames=len(frames), interval=60, blit=False)
plt.close()

HTML(anim.to_jshtml())