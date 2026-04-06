# REVISADO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from enum import Enum, auto
import sys

# ----------------------------------------------------------------------
# Clases y funciones del FEM 1D evolutivo (originales, ligeramente modificadas)
# ----------------------------------------------------------------------

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
        return (1.0 / self.h) * np.array([[1.0, -1.0], [-1.0, 1.0]])

    def matriz_masa_local(self):
        M = np.zeros((2, 2))
        N = [FuncionesForma.N1, FuncionesForma.N2]
        for a in range(2):
            for b in range(2):
                M[a, b] = self.h * integrar_gauss_2p(lambda e: N[a](e) * N[b](e))
        return M

    def vector_cargas_local(self, f):
        b = np.zeros(2)
        b[0] = self.h * integrar_gauss_2p(lambda e: f(self.mapear_a_fisico(e)) * FuncionesForma.N1(e))
        b[1] = self.h * integrar_gauss_2p(lambda e: f(self.mapear_a_fisico(e)) * FuncionesForma.N2(e))
        return b

class ProblemaFEM1DEvolutivo:
    def __init__(self, malla, f, c):
        self.malla = malla
        self.f = f
        self.c = c
        n = malla.n_nodos
        self.A = np.zeros((n, n))   # rigidez + reacción
        self.M = np.zeros((n, n))   # masa
        self.B = np.zeros(n)
        self.u = None

    def ensamblar(self):
        for i, j in self.malla.elementos:
            x1, x2 = self.malla.coordenadas_elemento(i, j)
            elem = Elemento1D(x1, x2)
            K = elem.matriz_rigidez_local()
            M = elem.matriz_masa_local()
            b = elem.vector_cargas_local(self.f)

            self.A[np.ix_([i, j], [i, j])] += self.c * K
            self.M[np.ix_([i, j], [i, j])] += M
            self.B[[i, j]] += b

    @staticmethod
    def aplicar_cc(A, B, condiciones):
        A = A.copy()
        B = B.copy()
        for nodo, valor, tipo in condiciones:
            if tipo == TipoCondicion.ESENCIAL:
                A[nodo, :] = 0
                A[nodo, nodo] = 1
                B[nodo] = valor
            elif tipo == TipoCondicion.NATURAL:
                B[nodo] += valor
        return A, B

    def resolver(self, u0, dt, n_pasos, condiciones, guardar_todos=False):
        """
        Resuelve la evolución temporal.
        Si guardar_todos=True, almacena la solución en cada paso y la devuelve.
        Si guardar_todos=False (por defecto), solo almacena cada 5 pasos para ahorrar memoria.
        """
        u = u0.copy()
        soluciones = [u.copy()]
        tiempos = [0.0]

        for paso in range(n_pasos):
            A_t = self.M + dt * self.A
            B_t = self.M @ u + dt * self.B
            A_t, B_t = self.aplicar_cc(A_t, B_t, condiciones)
            u = np.linalg.solve(A_t, B_t)

            if guardar_todos:
                soluciones.append(u.copy())
                tiempos.append((paso + 1) * dt)
            else:
                if (paso + 1) % 5 == 0:
                    soluciones.append(u.copy())
                    tiempos.append((paso + 1) * dt)

        self.u = u
        return u, soluciones, tiempos

    def imprimir(self):
        for i, x in enumerate(self.malla.nodos):
            print(f"u({x:.4f}) = {self.u[i]:.6f}")


# ----------------------------------------------------------------------
# Función para exportar el video
# ----------------------------------------------------------------------

def exportar_video(malla, soluciones, tiempos, archivo_salida="evolucion.mp4", fps=20, formato='gif'):
    """
    Exporta un video de la evolución temporal de la solución.

    Parámetros:
        malla: objeto Malla1D
        soluciones: lista de arrays con la solución en cada instante
        tiempos: lista de tiempos correspondientes
        archivo_salida: nombre del archivo de salida
        fps: frames por segundo
        formato: 'mp4' (requiere ffmpeg) o 'gif' (requiere Pillow)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = malla.nodos

    # Límites fijos del eje Y (para que no varíe la escala)
    y_min = min(np.min(sol) for sol in soluciones)
    y_max = max(np.max(sol) for sol in soluciones)
    margen = 0.1 * (y_max - y_min) if (y_max - y_min) != 0 else 0.1
    ax.set_ylim(y_min - margen, y_max + margen)
    ax.set_xlim(malla.a, malla.b)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.grid(True)

    # Línea inicial
    line, = ax.plot(x, soluciones[0], 'b-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

    def update(frame):
        line.set_ydata(soluciones[frame])
        time_text.set_text(f't = {tiempos[frame]:.3f}')
        return line, time_text

    anim = FuncAnimation(fig, update, frames=len(soluciones), interval=1000/fps, blit=True)

    try:
        if formato == 'mp4':
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='FEM'), bitrate=1800)
            anim.save(archivo_salida, writer=writer)
        elif formato == 'gif':
            writer = PillowWriter(fps=fps)
            anim.save(archivo_salida, writer=writer)
        else:
            raise ValueError("Formato no soportado. Use 'mp4' o 'gif'.")
        print(f"Video guardado como {archivo_salida}")
    except Exception as e:
        print(f"Error al guardar el video: {e}")
        print("Asegúrate de tener instalado ffmpeg para MP4 o Pillow para GIF.")
    finally:
        plt.close(fig)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":

    # Parámetros del problema
    a = 0.0
    b = 1.0
    n_elem = 100          # número de elementos
    c = 1.0               # coeficiente de difusión
    f = lambda x: np.cos(2 * np.pi * x)   # término fuente

    # Construcción de la malla
    malla = Malla1D(a, b, n_elem)

    # Creación del problema FEM
    problema = ProblemaFEM1DEvolutivo(malla, f, c)
    problema.ensamblar()

    # Condición inicial
    u0 = np.cos(2 * np.pi * malla.nodos)

    # Condiciones de contorno (Dirichlet en ambos extremos)
    condiciones = [
        (0, 1.0, TipoCondicion.ESENCIAL),
        (malla.n_nodos - 1, 1.0, TipoCondicion.ESENCIAL)
    ]

    # Parámetros temporales
    dt = 0.01
    n_pasos = 100

    # Resolver guardando TODOS los pasos (para el video fluido)
    u_final, soluciones, tiempos = problema.resolver(
        u0, dt, n_pasos, condiciones, guardar_todos=True
    )

    print("\nSolución final (primeros y últimos nodos):")
    problema.imprimir()

    # Exportar video (MP4)
    # Si no tienes ffmpeg, cambia formato='gif' y la extensión a .gif
    exportar_video(malla, soluciones, tiempos, archivo_salida="evolucion_fem.gif", fps=20, formato='gif')

    # Opcional: mostrar una gráfica con algunas curvas
    plt.figure(figsize=(8,5))
    for i in range(0, len(soluciones), max(1, len(soluciones)//10)):
        plt.plot(malla.nodos, soluciones[i], label=f"t={tiempos[i]:.2f}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Evolución temporal (selección de instantes)")
    plt.grid(True)
    plt.legend()
    plt.show()