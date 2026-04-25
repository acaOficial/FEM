# REVISADO

import numpy as np
from enum import Enum, auto


class TipoCondicion(Enum):
    ESENCIAL = auto()   # Dirichlet: fija el valor de u en el nodo
    NATURAL = auto()    # Neumann: aporta al vector del segundo miembro


def integrar_gauss_2p(func):
    # Puntos de Gauss en [0,1]
    puntos = np.array([
        (1.0 - 1.0 / np.sqrt(3.0)) / 2.0,
        (1.0 + 1.0 / np.sqrt(3.0)) / 2.0
    ])

    # Pesos de Gauss en [0,1]
    pesos = np.array([0.5, 0.5])

    # Aproximación de la integral
    resultado = 0.0
    for punto, peso in zip(puntos, pesos):
        resultado += peso * func(punto)

    return resultado


class FuncionesForma:
    @staticmethod
    def N1(e):
        # Función de forma asociada al nodo izquierdo
        return 1.0 - e

    @staticmethod
    def N2(e):
        # Función de forma asociada al nodo derecho
        return e

    @staticmethod
    def dN1_de(e):
        # Derivada de N1 respecto a la coordenada local e
        return -1.0

    @staticmethod
    def dN2_de(e):
        # Derivada de N2 respecto a la coordenada local e
        return 1.0


class Malla1D:
    def __init__(self, a, b, n_elementos):
        self.a = a
        self.b = b
        self.n_elementos = n_elementos

        # Nodos de la malla
        self.nodos = np.linspace(a, b, n_elementos + 1)

        # Conectividad: cada elemento une dos nodos consecutivos
        self.elementos = [(i, i + 1) for i in range(n_elementos)]

    @property
    def h(self):
        # Tamaño uniforme de elemento
        return (self.b - self.a) / self.n_elementos

    @property
    def n_nodos(self):
        return self.nodos.size

    def coordenadas_elemento(self, indice_i, indice_j):
        # Devuelve coordenadas físicas de los nodos de un elemento
        return self.nodos[indice_i], self.nodos[indice_j]


class Elemento1D:
    def __init__(self, x_izq, x_der):
        self.x_izq = x_izq
        self.x_der = x_der

    @property
    def h(self):
        # Longitud del elemento
        return self.x_der - self.x_izq

    def mapear_a_fisico(self, e):
        # Transformación del elemento de referencia [0,1]
        # al elemento físico [x_izq, x_der]
        return self.x_izq + self.h * e

    def de_dx(self):
        # Como x = x_izq + h*e, entonces de/dx = 1/h
        return 1.0 / self.h

    def dN_dx(self):
        # Derivadas de las funciones de forma respecto a x
        return np.array([
            -1.0 / self.h,
             1.0 / self.h
        ], dtype=float)

    def matriz_rigidez_local(self):
        # Matriz local del término de rigidez:
        # integral (dNi/dx)(dNj/dx) dx
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
        # Vector local del segundo miembro:
        # integral f(x)*Ni dx
        b_local = np.zeros(2, dtype=float)

        # Primer componente: asociado al nodo izquierdo del elemento
        b_local[0] = self.h * integrar_gauss_2p(
            lambda e: f(self.mapear_a_fisico(e)) * FuncionesForma.N1(e)
        )

        # Segundo componente: asociado al nodo derecho del elemento
        b_local[1] = self.h * integrar_gauss_2p(
            lambda e: f(self.mapear_a_fisico(e)) * FuncionesForma.N2(e)
        )

        return b_local


class ProblemaFEM1D:
    def __init__(self, malla, f, c):
        self.malla = malla
        self.f = f
        self.c = c

        # Matriz global del sistema
        self.A = np.zeros((malla.n_nodos, malla.n_nodos), dtype=float)

        # Vector global del segundo miembro
        self.B = np.zeros(malla.n_nodos, dtype=float)

        # Solución nodal
        self.u = None

    def ensamblar_matriz_global(self):
        # Recorremos todos los elementos de la malla
        for i, j in self.malla.elementos:
            # Coordenadas físicas del elemento actual
            x_izq, x_der = self.malla.coordenadas_elemento(i, j)

            # Construimos el elemento local
            elemento = Elemento1D(x_izq, x_der)

            # Parte de rigidez
            K_local = elemento.matriz_rigidez_local()

            # Parte de reacción
            M_local = elemento.matriz_reaccion_local(self.c)

            # Matriz local total
            A_local = K_local + M_local

            # Ensamblaje en la matriz global
            self.A[np.ix_([i, j], [i, j])] += A_local

    def ensamblar_vector_global(self):
        # Recorremos todos los elementos
        for i, j in self.malla.elementos:
            # Coordenadas físicas del elemento actual
            x_izq, x_der = self.malla.coordenadas_elemento(i, j)

            # Construimos el elemento local
            elemento = Elemento1D(x_izq, x_der)

            # Vector local del segundo miembro
            b_local = elemento.vector_cargas_local(self.f)

            # Ensamblaje en el vector global
            self.B[[i, j]] += b_local

    def ensamblar_sistema(self):
        # Montaje completo del sistema A*u = B
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
        # Resolvemos el sistema lineal
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
    # Parámetros del problema
    a = 0.0
    b = 10.0
    n_elementos = 100
    c = 1.0
    f = lambda x: 2.0 * x

    # Construcción de la malla
    malla = Malla1D(a, b, n_elementos)

    print("Nodos:", malla.nodos)
    print("Elementos:", malla.elementos)
    print("h global =", malla.h)
    print()

    # Construcción del problema FEM
    problema = ProblemaFEM1D(malla, f, c)

    # Ensamblaje de matriz y vector globales
    problema.ensamblar_sistema()

    print("Matriz global A:")
    print(problema.A)
    print()

    print("Vector global B:")
    print(problema.B)
    print()

    # Condiciones de contorno:
    # u(0) = 0  -> esencial
    # u'(10) = 1 -> natural
    condiciones = [
        (0, 0.0, TipoCondicion.ESENCIAL),
        (malla.n_nodos - 1, 1.0, TipoCondicion.NATURAL)
    ]

    # Aplicamos condiciones de contorno
    problema.aplicar_condiciones_contorno(condiciones)

    print("Matriz A con condiciones de contorno:")
    print(problema.A)
    print()

    print("Vector B con condiciones de contorno:")
    print(problema.B)
    print()

    # Resolución del sistema
    problema.resolver()

    # Mostrar solución en los nodos
    problema.imprimir_solucion()