import numpy as np
import matplotlib.pyplot as plt

Nx, Ny = 80, 40
x = np.linspace(-5, 5, Nx + 2)
y = np.linspace(0, 2, Ny + 2)
hx = x[1] - x[0]
hy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

cx = 1/hx**2
cy = 1/hy**2
denom = 2*cx + 2*cy

u = np.zeros((Ny+2, Nx+2))
u[:, 0] = np.sin(2*np.pi*y)   
u[:, -1] = np.sin(2*np.pi*y)  
u[-1, :] = np.sin(np.pi*x)    
u[0, :] = 0.0                  

for it in range(1, 10001):
    u_old = u.copy()
    for j in range(1, Ny+1):
        for i in range(1, Nx+1):
            u[j, i] = (cx*(u[j, i+1] + u[j, i-1]) +
                       cy*(u[j+1, i] + u[j-1, i])) / denom
    if np.max(np.abs(u - u_old)) < 1e-6:
        print(f"Konvergerte etter {it} iterasjoner")
        break

# PLOT
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

cf = axes[0].contourf(X, Y, u, levels=50, cmap='viridis')
plt.colorbar(cf, ax=axes[0])
axes[0].set_title("Konturplott")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

ax3d = fig.add_subplot(122, projection='3d')
fig.delaxes(axes[1])
ax3d.plot_surface(X, Y, u, cmap='viridis', edgecolor='none')
ax3d.set_title("3D-overflate")
ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("u")

plt.tight_layout()
plt.show()
