import numpy as np
from enum import Enum, auto


class TipoCondicion(Enum):
    ESENCIAL = auto()   # Dirichlet
    NATURAL = auto()    # Neumann


def integrar_gauss_2p(func):
    puntos = np.array([
        (1.0 - 1.0 / np.sqrt(3.0)) / 2.0,
        (1.0 + 1.0 / np.sqrt(3.0)) / 2.0
    ])
    pesos = np.array([0.5, 0.5])

    resultado = 0.0
    for p, w in zip(puntos, pesos):
        resultado += w * func(p)
    return resultado


class FuncionesForma:
    @staticmethod
    def N1(e): return 1.0 - e

    @staticmethod
    def N2(e): return e

    @staticmethod
    def dN1_de(e): return -1.0

    @staticmethod
    def dN2_de(e): return 1.0


class Malla1D:
    def __init__(self, a, b, n_elementos):
        self.a = a
        self.b = b
        self.n_elementos = n_elementos
        self.nodos = np.linspace(a, b, n_elementos + 1)
        self.elementos = [(i, i + 1) for i in range(n_elementos)]

    @property
    def h(self):
        return (self.b - self.a) / self.n_elementos

    @property
    def n_nodos(self):
        return self.nodos.size

    def coordenadas_elemento(self, i, j):
        return self.nodos[i], self.nodos[j]


class Elemento1D:
    def __init__(self, x_izq, x_der):
        self.x_izq = x_izq
        self.x_der = x_der

    @property
    def h(self):
        return self.x_der - self.x_izq

    def mapear_a_fisico(self, e):
        return self.x_izq + self.h * e

    def matriz_rigidez_local(self):
        return (1.0 / self.h) * np.array([
            [1.0, -1.0],
            [-1.0, 1.0]
        ])

    def matriz_reaccion_local(self, c):
        M = np.zeros((2, 2))
        N = [FuncionesForma.N1, FuncionesForma.N2]

        for a in range(2):
            for b in range(2):
                M[a, b] = c * self.h * integrar_gauss_2p(
                    lambda e: N[a](e) * N[b](e)
                )
        return M

    def matriz_masa_local(self):
        M = np.zeros((2, 2))
        N = [FuncionesForma.N1, FuncionesForma.N2]

        for a in range(2):
            for b in range(2):
                M[a, b] = self.h * integrar_gauss_2p(
                    lambda e: N[a](e) * N[b](e)
                )
        return M

    def vector_cargas_local(self, f):
        b = np.zeros(2)

        b[0] = self.h * integrar_gauss_2p(
            lambda e: f(self.mapear_a_fisico(e)) * FuncionesForma.N1(e)
        )

        b[1] = self.h * integrar_gauss_2p(
            lambda e: f(self.mapear_a_fisico(e)) * FuncionesForma.N2(e)
        )

        return b


class ProblemaFEM1DEvolutivo:
    def __init__(self, malla, f, c):
        self.malla = malla
        self.f = f
        self.c = c

        n = malla.n_nodos

        self.A = np.zeros((n, n))  # rigidez + reacción
        self.M = np.zeros((n, n))  # masa
        self.B = np.zeros(n)

        self.u = None

    def ensamblar(self):
        for i, j in self.malla.elementos:
            x1, x2 = self.malla.coordenadas_elemento(i, j)
            elem = Elemento1D(x1, x2)

            K = elem.matriz_rigidez_local()
            R = elem.matriz_reaccion_local(self.c)
            M = elem.matriz_masa_local()
            b = elem.vector_cargas_local(self.f)

            self.A[np.ix_([i, j], [i, j])] += K + R
            self.M[np.ix_([i, j], [i, j])] += M
            self.B[[i, j]] += b

    def aplicar_cc(self, A, B, condiciones):
        A = A.copy()
        B = B.copy()

        for nodo, valor, tipo in condiciones:
            if tipo == TipoCondicion.ESENCIAL:
                A[nodo, :] = 0
                A[:, nodo] = 0
                A[nodo, nodo] = 1
                B[nodo] = valor

            elif tipo == TipoCondicion.NATURAL:
                B[nodo] += valor

        return A, B

    def resolver(self, u0, dt, n_pasos, condiciones):
        u = u0.copy()

        for paso in range(n_pasos):
            A_t = self.M / dt + self.A
            B_t = self.B + (self.M / dt) @ u

            A_t, B_t = self.aplicar_cc(A_t, B_t, condiciones)

            u = np.linalg.solve(A_t, B_t)

        self.u = u
        return u

    def imprimir(self):
        for i, x in enumerate(self.malla.nodos):
            print(f"u({x}) = {self.u[i]}")


# ===================== MAIN =====================

if __name__ == "__main__":

    a = 0.0
    b = 10.0
    n_elem = 50

    c = 1.0
    f = lambda x: 2 * x

    malla = Malla1D(a, b, n_elem)

    problema = ProblemaFEM1DEvolutivo(malla, f, c)
    problema.ensamblar()

    # condición inicial
    u0 = np.zeros(malla.n_nodos)

    # condiciones de contorno
    condiciones = [
        (0, 0.0, TipoCondicion.ESENCIAL),
        (malla.n_nodos - 1, 1.0, TipoCondicion.NATURAL)
    ]

    # parámetros temporales
    dt = 0.1
    n_pasos = 50

    u_final = problema.resolver(u0, dt, n_pasos, condiciones)

    print("\nSolución final:")
    problema.imprimir()