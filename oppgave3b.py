import numpy as np
import matplotlib.pyplot as plt

# --- Problemdata ---
a, b = -1.0, 1.0          # intervall
ua, ub = 0.0, 2.0         # randbetingelser u(a)=ua, u(b)=ub
N = 80                    # antall indre punkter (øke for bedre nøyaktighet)

# --- Rutenett ---
x = np.linspace(a, b, N + 2)   # inkluderer randpunktene
h = x[1] - x[0]
xi = x[1:-1]                   # indre punkter

# --- Bygg matrise for u'' ~ (u_{i-1} - 2u_i + u_{i+1})/h^2 ---
main = (-2.0 / h**2) * np.ones(N)
off  = ( 1.0 / h**2) * np.ones(N - 1)
A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

# --- Høyreside f(x)=cos(pi x), med justering for randbetingelser ---
f = np.cos(np.pi * xi)
bvec = f.copy()
bvec[0]  -= ua / h**2
bvec[-1] -= ub / h**2

# --- Løs for indre u-verdier ---
u_int = np.linalg.solve(A, bvec)

# --- Sett sammen full løsning (inkl. randpunkter) ---
u_num = np.empty(N + 2)
u_num[0] = ua
u_num[-1] = ub
u_num[1:-1] = u_int

# --- Analytisk løsning fra (a): u(x) = x + 1 - (cos(pi x)+1)/pi^2 ---
u_exact = x + 1.0 - (1/np.pi**2)*(np.cos(np.pi * x) + 1.0)

# --- Feil ---
max_err = np.max(np.abs(u_num - u_exact))
rmse = np.sqrt(np.mean((u_num - u_exact)**2))
print(f"max|feil| = {max_err:.3e}")
print(f"RMSE      = {rmse:.3e}")

# --- Plot ---
plt.figure(figsize=(7, 4))
plt.plot(x, u_exact, label="Analytisk løsning")
plt.plot(x, u_num, "--", label="Numerisk (finitt differanse)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(f"Sammenligning: analytisk vs numerisk (N={N} indre punkter)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()