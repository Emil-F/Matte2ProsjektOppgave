import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Lukker gamle plott
plt.close("all")

vmax = 22
umax = 1/5
ps = [1, 2, 5]

# ------------------------------------------------------------
# Fluks for trafikk ligning: u_t + (J(u))_x = 0, der J(u) = u * v(u)
# v(u) = v_max * (1 - (u/u_max)^p)
# ------------------------------------------------------------
def flux_burgers(u, p):
    u_safe = np.clip(u, 0, umax)  # Unngår at programmet ikke krasjer hvis u blir ustabil
    return u_safe * vmax * (1 - (u_safe/umax)**p)
    

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


# Randbetingelser (Dirichlet)
def u_left(t):
    """Venstre randverdi u(a,t)."""
    return 0.0

def u_right(t):
    """Høyre randverdi u(b,t)."""
    return 0.0

# Initialbetingelse u(x,0)
u0 = np.piecewise(x_grid, [x_grid <= 0, x_grid > 0], [lambda x: umax, lambda x: 0])

# CFL-sjekk
cfl = (dt / dx) * np.max(np.abs(u0))
print("max|u0| =", np.max(np.abs(u0)))
print("CFL ~ (dt/dx)*max|u0| =", cfl)

# Lagrer u0 løsningen for alle tider
us = np.zeros((len(ps), t_grid.size, x_grid.size))
us[:, 0, :] = u0

# Setter verdier for u i alle posijoner og tidspunkter for hver verdi av p
for index, u in enumerate(us):
    for n_step in range(1, t_grid.size):
        t_now = t_grid[n_step - 1]
        u[n_step, :] = lax_friedrichs_step(
            flux_burgers, u[n_step - 1, :], ps[index], dx, dt, t_now, u_left, u_right
        )

# ------------------------------------------------------------
# Animasjon
# ------------------------------------------------------------

# Finner hastigheten til en bil ut ifra tettheten
def v(u, p):
    u_safe = np.clip(u, 0, umax)  # Unngår at programmet ikke krasjer hvis u blir ustabil
    return vmax * (1 - (u_safe/umax)**p)

def u_at_position(u_row, x):
    """Tetthet u(x,t) ved posisjon x ved lineær interpolasjon på x_grid."""
    return np.interp(x, x_grid, u_row)


fig, (axP, axV) = plt.subplots(2, 1, figsize=(6, 6))
colors = ["g", "b", "r"]

# Plotter et subplot for hver verdi av p
for i in range(len(ps)):
    P = np.zeros(t_grid.size)
    V = np.zeros(t_grid.size)

    P[0] = -900

    # Finner verdiene for P (posisjon) og V (hastighet)
    for n in range(1, t_grid.size):
        P[n] = P[n-1] + dt * V[n-1]
        u_point = u_at_position(us[i, n, :], P[n])
        V[n] = v(u_point, ps[i])

    # Faste aksegrenser
    axP.set_xlim(t_grid[0], t_grid[-1])
    axP.set_ylim(np.min(P), 1000)

    axP.set_xlabel("t")
    axP.set_ylabel("P(t)")
    axP.set_title("Numerisk løsning av P(t)")

    # Rød stiplet: startprofil
    axP.plot(t_grid, P, label=f"P(t) for p={ps[i]}", color=colors[i])


    # Faste aksegrenser
    axV.set_xlim(t_grid[0], t_grid[-1])
    axV.set_ylim(0, 40)

    axV.set_xlabel("t")
    axV.set_ylabel("V(t)")
    axV.set_title("Numerisk løsning av V(t)")

    axV.plot(t_grid, V, label=f"V(t) for p={ps[i]}", color=colors[i])

axP.legend()
axV.legend()

fig.tight_layout()
plt.show()