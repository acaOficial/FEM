# REVISADO

import numpy as np
from enum import Enum, auto

# ------------------------------------------------------------
# Tipos de condición de contorno
# ------------------------------------------------------------
class TipoCondicion(Enum):
    DIRICHLET = auto()
    NEUMANN = auto()

# ------------------------------------------------------------
# Integración numérica en el triángulo de referencia (3 puntos)
# ------------------------------------------------------------
def integrar_triangulo_3p(func):
    """
    Integra una función sobre el triángulo de referencia de vértices (0,0), (1,0), (0,1)
    usando la regla de 3 puntos de Gauss (orden 2).
    """
    # Puntos de Gauss en coordenadas baricéntricas (xi, eta)
    puntos = [
        (1/6, 2/3),   # (1/6, 2/3) en (xi, eta)
        (1/6, 1/6),
        (2/3, 1/6)
    ]
    pesos = [1/6, 1/6, 1/6]

    resultado = 0.0
    for (xi, eta), w in zip(puntos, pesos):
        resultado += w * func(xi, eta)
    return resultado

# ------------------------------------------------------------
# Integración numérica en la arista de referencia [0,1] (2 puntos)
# ------------------------------------------------------------
def integrar_linea_2p(func):
    """
    Integra una función sobre el intervalo [0,1] usando la regla de 2 puntos de Gauss.
    """
    puntos = [
        (1.0 - 1.0/np.sqrt(3.0)) / 2.0,
        (1.0 + 1.0/np.sqrt(3.0)) / 2.0
    ]
    pesos = [0.5, 0.5]

    resultado = 0.0
    for p, w in zip(puntos, pesos):
        resultado += w * func(p)
    return resultado

# ------------------------------------------------------------
# Funciones de forma y sus derivadas en el triángulo de referencia
# ------------------------------------------------------------
class FuncionesFormaTriangulo:
    """
    Funciones de forma para el elemento triangular lineal de 3 nodos.
    El elemento de referencia tiene vértices: (0,0), (1,0), (0,1).
    """
    @staticmethod
    def N1(xi, eta):
        """Nodo 1 (vértice en (0,0))"""
        return 1.0 - xi - eta

    @staticmethod
    def N2(xi, eta):
        """Nodo 2 (vértice en (1,0))"""
        return xi

    @staticmethod
    def N3(xi, eta):
        """Nodo 3 (vértice en (0,1))"""
        return eta

    @staticmethod
    def grad_N1():
        """Gradiente de N1 respecto a (xi, eta)"""
        return np.array([-1.0, -1.0])

    @staticmethod
    def grad_N2():
        return np.array([1.0, 0.0])

    @staticmethod
    def grad_N3():
        return np.array([0.0, 1.0])

    @staticmethod
    def todas_funciones():
        """Devuelve lista de funciones de forma"""
        return [FuncionesFormaTriangulo.N1,
                FuncionesFormaTriangulo.N2,
                FuncionesFormaTriangulo.N3]

    @staticmethod
    def todos_gradientes():
        """Devuelve lista de gradientes (en coordenadas de referencia)"""
        return [FuncionesFormaTriangulo.grad_N1(),
                FuncionesFormaTriangulo.grad_N2(),
                FuncionesFormaTriangulo.grad_N3()]

# ------------------------------------------------------------
# Funciones de forma para la arista de referencia [0,1] (elemento lineal 1D)
# ------------------------------------------------------------
class FuncionesFormaArista:
    @staticmethod
    def N1(xi):
        return 1.0 - xi

    @staticmethod
    def N2(xi):
        return xi

    @staticmethod
    def todas_funciones():
        return [FuncionesFormaArista.N1, FuncionesFormaArista.N2]

# ------------------------------------------------------------
# Clase para la malla 2D
# ------------------------------------------------------------
class Malla2D:
    def __init__(self, nodes, elements, edges):
        """
        nodes: lista de tuplas (x, y) con coordenadas de cada nodo.
        elements: lista de tuplas (id_elemento, [id_nodo1, id_nodo2, id_nodo3])
                  donde los ids de nodos son 1-indexados (como en Gmsh).
        edges: lista de tuplas (id_arista, tag, [id_nodo1, id_nodo2])
        """
        self.nodes = nodes                # lista de (x,y)
        self.elements = elements          # lista de (id, [n1,n2,n3])
        self.edges = edges                # lista de (id, tag, [n1,n2])
        self.n_nodos = len(nodes)
        self.n_elementos = len(elements)

    def obtener_coordenadas_nodo(self, idx):
        """idx es 0-indexado"""
        return self.nodes[idx]

    def obtener_nodos_elemento(self, elem_idx):
        """Devuelve coordenadas de los 3 nodos del elemento (en orden local)"""
        _, node_ids = self.elements[elem_idx]
        # node_ids son 1-indexados -> pasar a 0-indexado
        coords = [self.nodes[i-1] for i in node_ids]
        return coords

    def obtener_aristas_por_tag(self, tag):
        """Devuelve lista de aristas que tienen una etiqueta (tag) dada"""
        return [edge for edge in self.edges if edge[1] == tag]

    def obtener_aristas_por_tipo(self, tipo_condicion, mapper):
        """
        mapper: diccionario que asigna tag -> TipoCondicion
        """
        aristas = []
        for edge in self.edges:
            tag = edge[1]
            if tag in mapper and mapper[tag] == tipo_condicion:
                aristas.append(edge)
        return aristas

# ------------------------------------------------------------
# Clase para un elemento finito triangular lineal
# ------------------------------------------------------------
class Elemento2D:
    def __init__(self, coords_nodos):
        """
        coords_nodos: lista de 3 tuplas (x,y) ordenadas según el elemento local:
                      1: (0,0), 2: (1,0), 3: (0,1) en el de referencia.
        """
        self.coords = np.array(coords_nodos)  # shape (3,2)

    def jacobiano(self):
        """Calcula la matriz jacobiana de la transformación (xi,eta) -> (x,y)"""
        # Coordenadas de los nodos
        x1, y1 = self.coords[0]
        x2, y2 = self.coords[1]
        x3, y3 = self.coords[2]
        J = np.array([[x2 - x1, x3 - x1],
                      [y2 - y1, y3 - y1]])
        return J

    def det_jacobiano(self):
        J = self.jacobiano()
        det = np.linalg.det(J)
        if det <= 0:
            raise ValueError("Elemento con orientación incorrecta (determinante negativo)")
        return det

    def matriz_rigidez_local(self):
        """
        Matriz local del término de rigidez: integral (grad Ni · grad Nj) dOmega
        """
        J = self.jacobiano()
        detJ = self.det_jacobiano()
        invJ = np.linalg.inv(J)
        # Gradientes en coordenadas físicas: grad_phi = invJ^T * grad_phi_hat
        grad_ref = FuncionesFormaTriangulo.todos_gradientes()  # lista de arrays (2,)
        grad_fis = [invJ.T @ g for g in grad_ref]

        K_local = np.zeros((3, 3))
        area = 0.5 * detJ   # área del triángulo físico
        for a in range(3):
            for b in range(3):
                K_local[a, b] = np.dot(grad_fis[a], grad_fis[b]) * area
        return K_local

    def matriz_reaccion_local(self, c):
        """
        Matriz local del término de reacción: integral c * Ni * Nj dOmega
        """
        if c == 0.0:
            return np.zeros((3, 3))
        detJ = self.det_jacobiano()
        area = 0.5 * detJ

        # La integral sobre el triángulo de Ni*Nj se calcula numéricamente
        # pero para elementos lineales existe fórmula exacta:
        # Para funciones de forma lineales, integral Ni*Nj dOmega = (area/12)*(1+delta_ij)
        # Sin embargo, usamos integración numérica por generalidad.
        M_local = np.zeros((3, 3))
        N_funcs = FuncionesFormaTriangulo.todas_funciones()

        def integrando(xi, eta, a, b):
            return N_funcs[a](xi, eta) * N_funcs[b](xi, eta)

        for a in range(3):
            for b in range(3):
                integral = integrar_triangulo_3p(
                    lambda xi, eta: integrando(xi, eta, a, b)
                )
                M_local[a, b] = c * integral * detJ   # detJ porque el integrando está en ref
        return M_local

    def vector_cargas_local(self, f):
        """
        Vector local del segundo miembro: integral f(x) * Ni dOmega
        f: función que recibe un punto (x,y) y devuelve un escalar.
        """
        b_local = np.zeros(3)
        N_funcs = FuncionesFormaTriangulo.todas_funciones()
        detJ = self.det_jacobiano()
        J = self.jacobiano()
        # Punto base (primer nodo) para la transformación afín
        base = self.coords[0]

        def mapear_a_fisico(xi, eta):
            """Transforma coordenadas de referencia (xi,eta) a físicas (x,y)"""
            return base + J @ np.array([xi, eta])

        for a in range(3):
            def integrando(xi, eta):
                punto_fis = mapear_a_fisico(xi, eta)
                return f(punto_fis) * N_funcs[a](xi, eta)
            integral = integrar_triangulo_3p(integrando)
            b_local[a] = integral * detJ
        return b_local

# ------------------------------------------------------------
# Clase para un elemento de arista (condiciones de Neumann)
# ------------------------------------------------------------
class ElementoArista1D:
    def __init__(self, nodo_izq, nodo_der, funcion_neumann):
        """
        nodo_izq, nodo_der: coordenadas (x,y) de los extremos de la arista.
        funcion_neumann: función que devuelve g_N(x) en la arista.
        """
        self.izq = np.array(nodo_izq)
        self.der = np.array(nodo_der)
        self.g = funcion_neumann
        self.longitud = np.linalg.norm(self.der - self.izq)

    def vector_cargas_local(self):
        """
        Calcula el vector local de contribución al segundo miembro:
        integral sobre la arista de g_N * Ni ds, donde Ni son funciones de forma lineales (1D).
        """
        b_local = np.zeros(2)
        N_funcs = FuncionesFormaArista.todas_funciones()

        def mapear_a_fisico(xi):
            return self.izq + xi * (self.der - self.izq)

        for a in range(2):
            def integrando(xi):
                punto = mapear_a_fisico(xi)
                return self.g(punto) * N_funcs[a](xi)
            integral = integrar_linea_2p(integrando)
            b_local[a] = integral * self.longitud
        return b_local

# ------------------------------------------------------------
# Clase que ensambla y resuelve el problema FEM
# ------------------------------------------------------------
class ProblemaFEM2D:
    def __init__(self, malla, f, c, boundary_mapper, boundary_values):
        """
        malla: objeto Malla2D.
        f: función fuente f(x,y).
        c: coeficiente constante (escalar) del término de reacción.
        boundary_mapper: diccionario {tag: TipoCondicion}
        boundary_values: diccionario {tag: funcion} para las condiciones.
        """
        self.malla = malla
        self.f = f
        self.c = c
        self.boundary_mapper = boundary_mapper
        self.boundary_values = boundary_values

        self.A = np.zeros((malla.n_nodos, malla.n_nodos))
        self.B = np.zeros(malla.n_nodos)
        self.u = None

    def ensamblar_matriz_global(self):
        """Ensambla la matriz global (rigidez + reacción)"""
        for elem_idx in range(self.malla.n_elementos):
            coords = self.malla.obtener_nodos_elemento(elem_idx)
            elem = Elemento2D(coords)
            K_local = elem.matriz_rigidez_local()
            M_local = elem.matriz_reaccion_local(self.c)
            A_local = K_local + M_local

            # Obtener índices globales de los nodos del elemento (0-indexados)
            _, node_ids = self.malla.elements[elem_idx]
            ids = [i-1 for i in node_ids]   # pasar a 0-indexado

            # Ensamblaje
            for a, ia in enumerate(ids):
                for b, ib in enumerate(ids):
                    self.A[ia, ib] += A_local[a, b]

    def ensamblar_vector_global(self):
        """Ensambla el vector global del segundo miembro (término fuente)"""
        for elem_idx in range(self.malla.n_elementos):
            coords = self.malla.obtener_nodos_elemento(elem_idx)
            elem = Elemento2D(coords)
            b_local = elem.vector_cargas_local(self.f)

            _, node_ids = self.malla.elements[elem_idx]
            ids = [i-1 for i in node_ids]
            for a, ia in enumerate(ids):
                self.B[ia] += b_local[a]

    def aplicar_condiciones_contorno(self):
        """
        Aplica condiciones de contorno:
        - Dirichlet: fija el valor en el nodo (anula fila/columna y pone 1 en diagonal)
        - Neumann: añade contribución al vector B sobre las aristas correspondientes.
        """
        # Procesar condiciones de Neumann primero (aportan a B)
        aristas_neumann = self.malla.obtener_aristas_por_tipo(TipoCondicion.NEUMANN,
                                                              self.boundary_mapper)
        for edge in aristas_neumann:
            _, tag, node_ids = edge   # node_ids son 1-indexados
            # Obtener función g_N
            g_func = self.boundary_values.get(tag)
            if g_func is None:
                continue
            # Coordenadas de los dos nodos de la arista
            n1 = node_ids[0] - 1
            n2 = node_ids[1] - 1
            coord1 = self.malla.obtener_coordenadas_nodo(n1)
            coord2 = self.malla.obtener_coordenadas_nodo(n2)
            elem_arista = ElementoArista1D(coord1, coord2, g_func)
            b_local = elem_arista.vector_cargas_local()
            # Ensamblar en B
            self.B[n1] += b_local[0]
            self.B[n2] += b_local[1]

        # Procesar condiciones de Dirichlet (modifican A y B)
        aristas_dirichlet = self.malla.obtener_aristas_por_tipo(TipoCondicion.DIRICHLET,
                                                                self.boundary_mapper)
        # Recolectar nodos con Dirichlet (puede haber repeticiones)
        nodos_dirichlet = set()
        for edge in aristas_dirichlet:
            _, tag, node_ids = edge
            for nid in node_ids:
                nodos_dirichlet.add(nid - 1)   # 0-indexado

        # Para cada nodo Dirichlet, fijar el valor según la función correspondiente
        # Nota: Si un nodo pertenece a varias aristas con distintas funciones,
        # se toma la primera encontrada. En mallas bien definidas no debería ocurrir.
        for nodo in nodos_dirichlet:
            # Buscar la función g asociada a este nodo (podría venir de cualquiera de sus aristas)
            g_val = None
            for edge in aristas_dirichlet:
                _, tag, node_ids = edge
                if nodo+1 in node_ids:
                    g_func = self.boundary_values.get(tag)
                    if g_func is not None:
                        g_val = g_func(self.malla.obtener_coordenadas_nodo(nodo))
                        break
            if g_val is None:
                continue

            # Aplicar condición esencial
            self.B -= self.A[:, nodo] * g_val

            self.A[nodo, :] = 0.0
            self.A[:, nodo] = 0.0
            self.A[nodo, nodo] = 1.0
            self.B[nodo] = g_val

    def ensamblar_sistema(self):
        """Construye el sistema lineal A*u = B completo (sin condiciones de contorno)"""
        self.ensamblar_matriz_global()
        self.ensamblar_vector_global()

    def resolver(self):
        """Resuelve el sistema lineal después de aplicar condiciones de contorno"""
        self.u = np.linalg.solve(self.A, self.B)
        return self.u

# ------------------------------------------------------------
# Funciones de lectura de malla en formato Gmsh .msh (versión 2.2)
# ------------------------------------------------------------
def leer_malla_gmsh(archivo_msh):
    """
    Lee un archivo .msh (Gmsh) y devuelve nodos, elementos (triángulos) y aristas de contorno.
    Retorna: (nodes, elements, edges)
        nodes: lista de (x, y)
        elements: lista de (id_elemento, [n1, n2, n3]) (ids 1-indexados)
        edges: lista de (id_arista, tag, [n1, n2]) (ids 1-indexados)
    """
    with open(archivo_msh, 'r') as f:
        lines = f.readlines()

    # Buscar sección $Nodes
    nodes = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "$Nodes":
            i += 1
            num_nodes = int(lines[i].strip())
            i += 1
            for _ in range(num_nodes):
                parts = lines[i].strip().split()
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                # z = float(parts[3])  # ignoramos z
                nodes.append((x, y))
                i += 1
            break
        i += 1

    # Buscar sección $Elements
    elements = []
    edges = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "$Elements":
            i += 1
            num_elements = int(lines[i].strip())
            i += 1
            for _ in range(num_elements):
                parts = lines[i].strip().split()
                elem_id = int(parts[0])
                elem_type = int(parts[1])
                if elem_type == 1:   # línea (arista)
                    # Formato: id tipo num_tags tag ... nodos
                    num_tags = int(parts[2])
                    tag = int(parts[3])  # asumimos que el primer tag es el de condición
                    node1 = int(parts[3 + num_tags])
                    node2 = int(parts[4 + num_tags])
                    edges.append((elem_id, tag, [node1, node2]))
                elif elem_type == 2:   # triángulo lineal
                    num_tags = int(parts[2])
                    node1 = int(parts[3 + num_tags])
                    node2 = int(parts[4 + num_tags])
                    node3 = int(parts[5 + num_tags])
                    elements.append((elem_id, [node1, node2, node3]))
                i += 1
            break
        i += 1

    return nodes, elements, edges

# ------------------------------------------------------------
# Exportación de solución en formato similar al del colega
# ------------------------------------------------------------
def exportar_solucion(archivo_salida, malla, u):
    """Exporta la solución nodal a un archivo .inp con formato simple."""
    with open(archivo_salida, 'w') as f:
        # Cabecera
        f.write(f"{malla.n_nodos} {malla.n_elementos} 1 0 0\n")
        # Coordenadas
        for i, (x, y) in enumerate(malla.nodes, start=1):
            f.write(f"{i} {x} {y} 0.0\n")
        # Conectividad
        for elem_id, node_ids in malla.elements:
            n1, n2, n3 = node_ids
            f.write(f"{elem_id} 1 tri {n1} {n2} {n3}\n")
        # Cabecera de variable
        f.write("1 1\n")
        f.write("u, sol\n")
        # Valores nodales
        for i, valor in enumerate(u, start=1):
            f.write(f"{i} {valor}\n")

# ------------------------------------------------------------
# Ejemplo de uso (main)
# ------------------------------------------------------------
if __name__ == "__main__":
    # Archivos de entrada/salida
    mesh_file = "malla.msh"
    output_file = "solucion_aca.inp"

    # Definir mapeo de etiquetas y funciones de contorno (igual que en el código del colega)
    BoundaryMapper = {
        5: TipoCondicion.DIRICHLET,
        6: TipoCondicion.NEUMANN
    }
    BoundaryValues = {
        5: lambda x: np.exp(-10 * (x[0]-0.5)**2 - 10 * (x[1]-0.5)**2),
        6: lambda x: -20*(x[1]-0.5)*np.exp(-10*((x[0]-0.5)**2 + (x[1]-0.5)**2))
    }

    # Parámetros de la EDP
    c = 0.0
    f = lambda x: -80*np.exp(-5 * (1-2*x[0]+2*x[0]**2-2*x[1]+2*x[1]**2)) * (2-5*x[0]+5*x[0]**2-5*x[1]+5*x[1]**2)

    # Leer malla
    nodes, elements, edges = leer_malla_gmsh(mesh_file)
    malla = Malla2D(nodes, elements, edges)

    # Crear problema FEM
    problema = ProblemaFEM2D(malla, f, c, BoundaryMapper, BoundaryValues)

    # Ensamblar sistema
    problema.ensamblar_sistema()

    # Aplicar condiciones de contorno (incluye Neumann y Dirichlet)
    problema.aplicar_condiciones_contorno()

    # Resolver
    u = problema.resolver()

    # Exportar solución
    exportar_solucion(output_file, malla, u)


    print("Solución FEM en los nodos:")
    for i, (x, y) in enumerate(malla.nodes):
        print(f"Nodo {i}: ({x:.3f}, {y:.3f}) -> u = {u[i]:.6e}")
        
    # Opcional: mostrar diferencias con solución exacta (si se conoce)
    # u_exacta = lambda x: np.exp(-10 * (x[0]-0.5)**2 - 10 * (x[1]-0.5)**2)
    # print("Diferencias en los nodos (aprox - exacta):")
    # for i, (x, y) in enumerate(malla.nodes):
    #     diff = u[i] - u_exacta((x, y))
    #     print(f"Nodo {i}: ({x:.3f}, {y:.3f}) -> {diff:.6e}")