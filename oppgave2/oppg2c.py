import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# ------------------------------------------------------------
# Oppsett for figurer (nyttig i notater for å unngå "gamle" plott)
# ------------------------------------------------------------
plt.close("all")
plt.figure(figsize=(6, 4))

vmax = 22
umax = 1/5
ps = [1, 2, 5]

# ------------------------------------------------------------
# Fluks for Burgers' ligning: u_t + (f(u))_x = 0, der f(u) = u^2/2
# ------------------------------------------------------------
def flux_burgers(u, p):
    """Fluksen f(u) for Burgers' ligning."""
    return u * vmax * (1 - (u/umax)**p)
    


# ------------------------------------------------------------
# Lax–Friedrichs-metoden (1D, eksplisitt)
# ------------------------------------------------------------
# fluks: funksjon f(u)
# u: array med løsningen ved tid t (på romgitteret)
# dx: steglengde i rom
# dt: steglengde i tid
# t: nåværende tid (brukes av randbetingelsene)
# u_left(t): randverdi på venstre kant (x = a)
# u_right(t): randverdi på høyre kant (x = b)
def lax_friedrichs_step(fluks, u, p, dx, dt, t, u_left, u_right):
    """Utfører ett tidssteg med Lax–Friedrichs."""
    n = u.size
    u_next = np.zeros_like(u)

    # Indre punkter (i = 1, ..., n-2)
    # Formelen:
    # u_i^{n+1} = 1/2 (u_{i-1}^n + u_{i+1}^n) - (dt/(2dx)) (f(u_{i+1}^n) - f(u_{i-1}^n))
    for i in range(1, n - 1):
        u_next[i] = 0.5 * (u[i - 1] + u[i + 1]) - (dt / (2 * dx)) * (
            fluks(u[i + 1], p) - fluks(u[i - 1], p)
        )

    # Randbetingelser (Dirichlet)
    u_next[0] = u_left(t)     # venstre rand
    u_next[-1] = u_right(t)   # høyre rand

    return u_next


# ------------------------------------------------------------
# Gitter i tid og rom
# ------------------------------------------------------------
T = 60         # sluttid
nt = 1000         # antall tidspunkter
a, b = -1000, 1000 # romintervall
nx = 100         # antall rompunkter

t_grid = np.linspace(0.0, T, nt)
x_grid = np.linspace(a, b, nx)

dt = t_grid[1] - t_grid[0]
dx = x_grid[1] - x_grid[0]





plt.close("all")

# ------------------------------------------------------------
# Vindu 2: Sett initialbetingelse, løs med Lax–Friedrichs, og animer
# Forutsetter at vindu 1 allerede er kjørt (x_grid, t_grid, dx, dt, osv.)
# ------------------------------------------------------------

# Randbetingelser (Dirichlet)
def u_left(t):
    """Venstre randverdi u(a,t)."""
    return 0.0

def u_right(t):
    """Høyre randverdi u(b,t)."""
    return 0.0

# Initialbetingelse u(x,0)
u0 = np.piecewise(x_grid, [x_grid <= 0, x_grid > 0], [lambda x: umax, lambda x: 0])

# CFL-sjekk (pedagogisk)
cfl = (dt / dx) * np.max(np.abs(u0))
print("max|u0| =", np.max(np.abs(u0)))
print("CFL ~ (dt/dx)*max|u0| =", cfl)

# Lagrer løsningen for alle tider (greit for animasjon)
us = np.zeros((len(ps), t_grid.size, x_grid.size))
us[:, 0, :] = u0

# Tidsløkken
for uIndex, u in enumerate(us):
    for n_step in range(1, t_grid.size):
        t_now = t_grid[n_step - 1]
        u[n_step, :] = lax_friedrichs_step(
            flux_burgers, u[n_step - 1, :], ps[uIndex], dx, dt, t_now, u_left, u_right
        )

# Diagnose: sjekk at løsningen faktisk endrer seg
print("Maks endring fra t0 til t1:", np.max(np.abs(us[0, 1, :] - us[0, 0, :])))

# ------------------------------------------------------------
# Animasjon
# ------------------------------------------------------------
fig, axs = plt.subplots(1, len(us), figsize=(16, 4))
lines = []

for i, ax in enumerate(axs.flatten()):
    # Faste aksegrenser (som i eksempelet)
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_ylim(np.min(us[i]), np.max(us[i]))

    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(f"Løsning med p={ps[i]}")

    # Rød stiplet: startprofil
    ax.plot(x_grid, u0, color="red", linestyle="--", label="u(x,0) (startprofil)")

    # Blå kurve: løsningen som oppdateres
    line, = ax.plot(x_grid, us[i, 0, :], color="blue", label="u(x,t)")
    lines.append(line)

    ax.legend()

def animate(i):
    for lineIndex, line in enumerate(lines):
        line.set_ydata(us[lineIndex, i, :])
    return lines

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=t_grid.size,
    interval=20,
    blit=True
)

HTML(ani.to_jshtml())
plt.show()