import numpy as np
import matplotlib.pyplot as plt

Nx = 100
x = np.linspace(-1, 1, Nx + 1)
dx = x[1] - x[0]
dt = 0.4 * dx**2
r = dt / dx**2
T = 0.5

f = np.cos(np.pi * x)
u = 1 + x + 5 * np.sin(np.pi * x)

u[0] = 0.0
u[-1] = 2.0

plot_tider = [0.0, 0.01, 0.05, 0.1, 0.5]
plottet = {t: False for t in plot_tider}

# LAGRE INITIALTILSTAND
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, u, label=f"t = 0.000")
plottet[0.0] = True

t = 0.0
n_steps = int(T / dt)

for n in range(n_steps):
    u_new = u.copy()
    u_new[1:-1] = u[1:-1] + r * (u[:-2] - 2*u[1:-1] + u[2:]) - dt * f[1:-1]
    u_new[0] = 0.0
    u_new[-1] = 2.0
    u = u_new
    t += dt

    for tp in plot_tider:
        if not plottet[tp] and abs(t - tp) < dt / 2:
            ax.plot(x, u, label=f"t = {t:.3f}")
            plottet[tp] = True

u_stat = (1/np.pi**2) * np.cos(np.pi * x) + x + 1 - 1/np.pi**2
ax.plot(x, u_stat, 'k--', linewidth=2, label="Stasjonær (analytisk)")

ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.set_title("Varmelikning med Forward Euler\n"
             r"$u_t = u_{xx} - \cos(\pi x)$, $u(x,0) = 1 + x + 5\sin(\pi x)$")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Parametere: Nx={Nx}, dx={dx:.4f}, dt={dt:.2e}, r={r:.2f}, T={T}")
print(f"Antall tidsskritt: {n_steps}")
