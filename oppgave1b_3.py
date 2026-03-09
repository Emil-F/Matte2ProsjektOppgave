import numpy as np
import matplotlib.pyplot as plt

# Konstanter, hvor maks hastighet er 30 m/s og maks tetther er lik 1
v_max = 30
u_max = 1

# Tetthet fra 0 til 1
u = np.linspace(0, 1, 500)

# Funksjon 1
v1 = v_max * (1 - (u / u_max))**2

# Funksjon 2
v2 = v_max * (1 - np.sqrt(u / u_max))

# Dette er plottet. Bruker figsize til å lage størrelsen på plottet. 
# Så legger vi inn funksjonene
plt.figure(figsize=(8, 6))
plt.plot(u, v1, label="V1")
plt.plot(u, v2, label="V2")

# Her setter vi inn navn og overskrifter. Samt lager et rutenett i bakgrunn.
plt.title("Sammenheng mellom trafikktetthet og bilfart")
plt.xlabel("Trafikktetthet")
plt.ylabel("Hastighet")
plt.grid(True)
plt.legend()
plt.show()