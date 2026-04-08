import numpy as np
import matplotlib.pyplot as plt
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


class Malla1D:
    def __init__(self, nodos):
        self.nodos = np.array(sorted(nodos), dtype=float)
        self.elementos = [(i, i + 1) for i in range(len(self.nodos) - 1)]

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

    def vector_cargas_local(self, f):
        b_local = np.zeros(2)

        b_local[0] = self.h * integrar_gauss_2p(
            lambda e: f(self.mapear_a_fisico(e)) * FuncionesForma.N1(e)
        )
        b_local[1] = self.h * integrar_gauss_2p(
            lambda e: f(self.mapear_a_fisico(e)) * FuncionesForma.N2(e)
        )

        return b_local

    def norma_L2_f(self, f):
        val = self.h * integrar_gauss_2p(
            lambda e: f(self.mapear_a_fisico(e))**2
        )
        return np.sqrt(val)

    def indicador_error(self, f):
        return self.h * self.norma_L2_f(f)


class ProblemaFEM1D:
    def __init__(self, malla, f):
        self.malla = malla
        self.f = f

        self.A = np.zeros((malla.n_nodos, malla.n_nodos))
        self.B = np.zeros(malla.n_nodos)
        self.u = None

    def ensamblar_matriz_global(self):
        for i, j in self.malla.elementos:
            x_i, x_j = self.malla.coordenadas_elemento(i, j)
            elem = Elemento1D(x_i, x_j)

            K = elem.matriz_rigidez_local()
            self.A[np.ix_([i, j], [i, j])] += K

    def ensamblar_vector_global(self):
        for i, j in self.malla.elementos:
            x_i, x_j = self.malla.coordenadas_elemento(i, j)
            elem = Elemento1D(x_i, x_j)

            b = elem.vector_cargas_local(self.f)
            self.B[[i, j]] += b

    def ensamblar_sistema(self):
        self.ensamblar_matriz_global()
        self.ensamblar_vector_global()

    def aplicar_condiciones_contorno(self, condiciones):
        for nodo, valor, tipo in condiciones:
            if tipo == TipoCondicion.ESENCIAL:
                self.A[nodo, :] = 0.0
                self.A[:, nodo] = 0.0
                self.A[nodo, nodo] = 1.0
                self.B[nodo] = valor
            else:
                self.B[nodo] += valor

    def resolver(self):
        self.u = np.linalg.solve(self.A, self.B)
        return self.u

    def calcular_errores(self):
        errores = []

        for i, j in self.malla.elementos:
            x_i, x_j = self.malla.coordenadas_elemento(i, j)
            elem = Elemento1D(x_i, x_j)

            eta = elem.indicador_error(self.f)
            errores.append(eta)

        return np.array(errores)

    def elementos_a_refinar(self, alpha=0.5):
        errores = self.calcular_errores()
        max_e = np.max(errores)

        indices = [i for i, e in enumerate(errores) if e > alpha * max_e]
        return indices, errores

    def refinar_malla(self, indices):
        nuevos_nodos = list(self.malla.nodos)

        for idx in indices:
            i, j = self.malla.elementos[idx]
            x_i = self.malla.nodos[i]
            x_j = self.malla.nodos[j]

            mid = 0.5 * (x_i + x_j)

            if mid not in nuevos_nodos:
                nuevos_nodos.append(mid)

        nuevos_nodos = np.array(sorted(nuevos_nodos))
        return Malla1D(nuevos_nodos)


def resolver_fem_1d(nodos, f, condiciones):
    """
    Resuelve el problema FEM en una malla dada y devuelve (x, u).
    """
    malla = Malla1D(nodos)
    problema = ProblemaFEM1D(malla, f)
    problema.ensamblar_sistema()
    problema.aplicar_condiciones_contorno(condiciones)
    u = problema.resolver()
    return malla.nodos.copy(), u.copy()


def adaptativo_1d(nodos_iniciales, f, alpha=0.5, max_iter=5):
    """
    Ejecuta refinamiento adaptativo y guarda el historial:
    historial[k] = (x_k, u_k)
    """
    historial = []
    malla = Malla1D(nodos_iniciales)

    for _ in range(max_iter):
        condiciones = [
            (0, 0.0, TipoCondicion.ESENCIAL),
            (malla.n_nodos - 1, 0.0, TipoCondicion.ESENCIAL)
        ]

        problema = ProblemaFEM1D(malla, f)
        problema.ensamblar_sistema()
        problema.aplicar_condiciones_contorno(condiciones)
        u = problema.resolver()

        historial.append((malla.nodos.copy(), u.copy()))

        indices, errores = problema.elementos_a_refinar(alpha=alpha)

        if len(indices) == 0:
            break

        malla = problema.refinar_malla(indices)

    return historial


# =========================
# MAIN + GRAFICA
# =========================

if __name__ == "__main__":
    a, b = 0.0, 1.0
    f = lambda x: np.exp(-100.0 * (x - 0.5)**2)

    # referencia "fina"
    x_ref = np.linspace(a, b, 2000)
    condiciones_ref = [
        (0, 0.0, TipoCondicion.ESENCIAL),
        (len(x_ref) - 1, 0.0, TipoCondicion.ESENCIAL)
    ]
    x_ref, u_ref = resolver_fem_1d(x_ref, f, condiciones_ref)

    # historial adaptativo
    historial = adaptativo_1d(
        nodos_iniciales=np.linspace(a, b, 3),
        f=f,
        alpha=0.5,
        max_iter=5
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    for k, (xk, uk) in enumerate(historial):
        ax = axes[k]

        # solución FEM de la iteración k
        ax.plot(xk, uk, '-o', markersize=4, label='FEM adaptativo')

        # referencia fina
        ax.plot(x_ref, u_ref, label='Referencia fina')

        # nodos de la malla en rojo sobre el eje x
        ax.plot(xk, np.zeros_like(xk), 'r.', markersize=6, label='Nodos')

        ax.set_title(f'Iteración {k}')
        ax.set_xlim(a, b)
        ax.grid(False)

    # apagar ejes sobrantes
    for j in range(len(historial), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()