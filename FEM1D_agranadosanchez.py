import numpy as np
from enum import Enum, auto


class TipoCondicion(Enum):
    ESENCIAL = auto()
    NATURAL = auto()


def integrar_gauss_2p(func):
    puntos = np.array([
        (1.0 - 1.0 / np.sqrt(3.0)) / 2.0,
        (1.0 + 1.0 / np.sqrt(3.0)) / 2.0
    ])

    pesos = np.array([0.5, 0.5])
    resultado = 0.0
    for punto, peso in zip(puntos, pesos):
        resultado += peso * func(punto)

    return resultado


class FuncionesForma:
    @staticmethod
    def N1(e):
        return 1.0 - e

    @staticmethod
    def N2(e):
        return e

    @staticmethod
    def dN1_de(e):
        return -1.0

    @staticmethod
    def dN2_de(e):
        return 1.0


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

    def coordenadas_elemento(self, indice_i, indice_j):
        return self.nodos[indice_i], self.nodos[indice_j]


class Elemento1D:
    def __init__(self, x_izq, x_der):
        self.x_izq = x_izq
        self.x_der = x_der

    @property
    def h(self):
        return self.x_der - self.x_izq

    def mapear_a_fisico(self, e):
        return self.x_izq + self.h * e

    def de_dx(self):
        return 1.0 / self.h

    def dN_dx(self):
        return np.array([
            -1.0 / self.h,
             1.0 / self.h
        ], dtype=float)

    def matriz_rigidez_local(self):
        return (1.0 / self.h) * np.array([
            [1.0, -1.0],
            [-1.0, 1.0]
        ], dtype=float)

    def matriz_reaccion_local(self, c):
        M_local = np.zeros((2, 2), dtype=float)

        N = [FuncionesForma.N1, FuncionesForma.N2]

        for a in range(2):
            for b in range(2):
                valor = c * self.h * integrar_gauss_2p(
                    lambda e: N[a](e) * N[b](e)
                )
                M_local[a, b] = valor

        return M_local
    
    def vector_cargas_local(self, f):
        b_local = np.zeros(2, dtype=float)

        b_local[0] = self.h * integrar_gauss_2p(
            lambda e: f(self.mapear_a_fisico(e)) * FuncionesForma.N1(e)
        )

        b_local[1] = self.h * integrar_gauss_2p(
            lambda e: f(self.mapear_a_fisico(e)) * FuncionesForma.N2(e)
        )

        return b_local


class ProblemaFEM1D:
    def __init__(self, malla, f, c):
        self.malla = malla
        self.f = f
        self.c = c

        self.A = np.zeros((malla.n_nodos, malla.n_nodos), dtype=float)
        self.B = np.zeros(malla.n_nodos, dtype=float)
        self.u = None

    def ensamblar_matriz_global(self):
        for i, j in self.malla.elementos:
            x_izq, x_der = self.malla.coordenadas_elemento(i, j)
            
            elemento = Elemento1D(x_izq, x_der)
            K_local = elemento.matriz_rigidez_local()
            M_local = elemento.matriz_reaccion_local(self.c)

            A_local = K_local + M_local
            self.A[np.ix_([i, j], [i, j])] += A_local

    def ensamblar_vector_global(self):
        for i, j in self.malla.elementos:
            x_izq, x_der = self.malla.coordenadas_elemento(i, j)
            elemento = Elemento1D(x_izq, x_der)
            b_local = elemento.vector_cargas_local(self.f)

            self.B[[i, j]] += b_local

    def ensamblar_sistema(self):
        self.ensamblar_matriz_global()
        self.ensamblar_vector_global()

    def aplicar_condiciones_contorno(self, condiciones):
        for nodo, valor, tipo in condiciones:

            if tipo == TipoCondicion.ESENCIAL:
                self.B -= self.A[:, nodo] * valor

                self.A[nodo, :] = 0.0
                self.A[:, nodo] = 0.0

                self.A[nodo, nodo] = 1.0
                self.B[nodo] = valor

            elif tipo == TipoCondicion.NATURAL:
                self.B[nodo] += valor
    
    def resolver(self):
        self.u = np.linalg.solve(self.A, self.B)
        return self.u

    def imprimir_solucion(self):
        if self.u is None:
            print("La solución todavía no se ha calculado.")
            return

        print("Solución nodal:")
        for i, x in enumerate(self.malla.nodos):
            print(f"u({x}) = {self.u[i]}")


if __name__ == "__main__":
    a = 0.0
    b = 10.0
    n_elementos = 100
    c = 1.0
    f = lambda x: 2.0 * x

    malla = Malla1D(a, b, n_elementos)

    print("Nodos:", malla.nodos)
    print("Elementos:", malla.elementos)
    print("h global =", malla.h)
    print()

    problema = ProblemaFEM1D(malla, f, c)
    problema.ensamblar_sistema()

    print("Matriz global A:")
    print(problema.A)
    print()

    print("Vector global B:")
    print(problema.B)
    print()

    condiciones = [
        (0, 0.0, TipoCondicion.ESENCIAL),
        (malla.n_nodos - 1, 1.0, TipoCondicion.NATURAL)
    ]

    problema.aplicar_condiciones_contorno(condiciones)

    print("Matriz A con condiciones de contorno:")
    print(problema.A)
    print()

    print("Vector B con condiciones de contorno:")
    print(problema.B)
    print()

    # Resolución del sistema
    problema.resolver()
    problema.imprimir_solucion()