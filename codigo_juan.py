#Ejercicio para la práctica
import matplotlib.pyplot as plt
import numpy as np

N = 100
L = 1.0
h = L / N

k = 1.0   # coeficiente difusión (antes era c)
u0 = 0.0  # Dirichlet en primer extremo
u0L = 1.0 # Dirichlet en ultimo extremo

# Funcion a resolver
def f(x):
    return np.cos(2*np.pi*x)

# funciones de forma en coordenada de referencia xi
def N1(xi):
  return (1.0 - xi)

def N2(xi):
  return xi

# Malla
x_nodes = np.linspace(0.0, L, N + 1)
n_nodes = N + 1

# Matriz y vector global
W = np.zeros((n_nodes, n_nodes))   # matriz de rigidez
M = np.zeros((n_nodes, n_nodes))   # matriz de masa
b = np.zeros(n_nodes)

# Cuadratura gauss
gauss_xi = np.array([(1-(1.0/np.sqrt(3.0)))/2,  (1+(1.0/np.sqrt(3.0)))/2])
gauss_w  = np.array([0.5, 0.5])

# derivadas respecto a x
dN_dx = np.array([-1.0, 1.0])


# Bucle de los elementos
for n in range(N):

    elem_nodes = np.array([n, n+1], dtype=int)

    x1 = x_nodes[n]
    x2 = x_nodes[n+1]
    he = x2 - x1

    We = np.zeros((2, 2))   # rigidez
    Me = np.zeros((2, 2))   # masa

    # Matriz de rigidez (con k)
    for alpha in range(2):
        for beta in range(2):
            We[alpha, beta] += k * dN_dx[alpha] * dN_dx[beta] * (1/he)

    # Matriz de masa
    for alpha in range(2):
        for beta in range(2):
            integ = 0.0
            for (xi, w) in zip(gauss_xi, gauss_w):
                Na = N1(xi) if alpha == 0 else N2(xi)
                Nb = N1(xi) if beta  == 0 else N2(xi)
                integ += w * Na * Nb
            Me[alpha, beta] += integ * he


    # Vector elemental
    be = np.zeros(2)

    for alpha in range(2):
        integ = 0.0
        for (xi, w) in zip(gauss_xi, gauss_w):
            x_gauss = x1 + he*xi
            Na = N1(xi) if alpha == 0 else N2(xi)
            integ += w * Na * f(x_gauss)
        be[alpha] += integ * he


    # Ensamblaje
    for alpha in range(2):
        Wglob = elem_nodes[alpha]
        b[Wglob] += be[alpha]

        for beta in range(2):
            Bglob = elem_nodes[beta]
            W[Wglob, Bglob] += We[alpha, beta]
            M[Wglob, Bglob] += Me[alpha, beta]


# Evaluacion en el tiempo
dt = 0.01
n_steps = 100
solutions = []
times = []

# condición inicial
u = x_nodes**2

solutions.append(u.copy())
times.append(0.0)

for step in range(n_steps):

    W_time = M + dt * W
    b_time = M @ u + dt * b

    # Dirichlet en ambos extremos
    W_time[0, :] = 0.0
    W_time[0, 0] = 1.0
    b_time[0] = u0

    W_time[-1, :] = 0.0
    W_time[-1, -1] = 1.0
    b_time[-1] = u0L

    u = np.linalg.solve(W_time, b_time)

    # guardar cada 10 pasos (para no saturar la gráfica)
    if step % 5 == 0:
        solutions.append(u.copy())
        times.append(step * dt)


# Resultados
print("\nSolución final u en nodos:")
for i in range(len(u)):
  print(u[i])


# Representacion grafica
x_dense = np.linspace(0.0, L, 2001)
u_dense = np.interp(x_dense, x_nodes, u)


# plt.figure()

# for i, u_sol in enumerate(solutions):
#     plt.plot(x_nodes, u_sol, label=f"t={times[i]:.2f}")

# plt.xlabel("x")
# plt.ylabel("u(x,t)")
# plt.title("Evolución temporal (FEM 1D calor)")
# plt.grid(True)
# plt.legend()
# plt.show()