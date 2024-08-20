import numpy as np
import sys 
sys.path.append('../data_structure')


try:
    from graph_datastructure import AdjList, AdjMatrix, Vertex, Edge
except ModuleNotFoundError:
    from data_structure.graph_datastructure import AdjList, AdjMatrix, Vertex, Edge


class Graph:
    def __init__(self, V, E, backend = 'VE'):
        for v in V:
            assert isinstance(v, Vertex) 
        for e in E:
            assert isinstance(e, Edge)
            assert e.from_vertex in V 
            assert e.to_vertex in V 

        self.V = V 
        self.E = E 
        self.backend = backend

        if backend == 'VE':
            pass 
        elif backend == 'adjacent_list': # key : vertex, value : edge
            self.adjacent_list = AdjList(V, E)
        elif backend == 'adjacnet_matrix':
            pass 
    
    
    def add_vertex(self, v):
        assert isinstance(v, Vertex)
        if self.backend == 'VE':
            self.V += [v] 
        elif self.backend == 'adjacent_list': 
            self.adjacent_list.add_vertex(v)    
        elif self.backend == 'adjacnet_matrix':
            pass 
    
    def remove_vertex(self, v):
        assert isinstance(v, Vertex)
        if self.backend == 'VE':
            # for 문을 도는 list에서 remove()를 하면 list 원본 자체에서 삭제하기 때문에 반복문 수행 시 index가 꼬이는 현상이 발생. 앞으로 그러지 말 것.
            edges_to_remove = []
            for e in self.E:
                if e.from_vertex == v or e.to_vertex == v:
                    # self.E.remove(e) 
                    edges_to_remove.append(e)
            for e in edges_to_remove:
                self.E.remove(e)
            self.V.remove(v)
        elif self.backend == 'adjacent_list': 
            print('before')
            print(self.adjacent_list.get_vertices())
            print(self.adjacent_list.get_edges())
            self.adjacent_list.remove_vertex(v) 
            print('after')
            print(self.adjacent_list.get_vertices())
            print(self.adjacent_list.get_edges())
        elif self.backend == 'adjacnet_matrix':
            pass 

    def add_edge(self, e):
        assert isinstance(e, Edge)
        if self.backend == 'VE':
            self.E += [e] 
        elif self.backend == 'adjacent_list': 
            self.adjacent_list.add_edge(e) 
        elif self.backend == 'adjacnet_matrix':
            pass 

    def remove_edge(self, e):
        assert isinstance(e, Edge)
        if self.backend == 'VE':
            self.E.remove(e) 
        elif self.backend == 'adjacent_list': 
            self.adjacent_list.remove_edge(e) 
        elif self.backend == 'adjacnet_matrix':
            pass 

    def get_vertices(self):
        vertex_list = []
        if self.backend == 'VE':
            for V in self.V:
                vertex_list.append(V.datum)
        elif self.backend == 'adjacent_list': 
            vertex_list = [x.datum for x in self.adjacent_list.get_vertices()]
        elif self.backend == 'adjacnet_matrix':
            pass 
        return vertex_list 

    def get_neighbors(self, v):
        assert isinstance(v, Vertex)
        neighbor_vertex = []
        if self.backend == 'VE':
            for e in self.E:
                if v == e.from_vertex:
                    neighbor_vertex.append(e.to_vertex.datum)
        elif self.backend == 'adjacent_list': 
            for k in self.adjacent_list.get_vertices():
                if v == k:
                    neighbor_vertex = self.adjacent_list.get_neighbors(k)
        elif self.backend == 'adjacnet_matrix':
            pass 
        return neighbor_vertex 

    def dfs(self, src):
        assert isinstance(src, Vertex)
        if self.backend == 'VE':
            pass 
        elif self.backend == 'adjacent_list': 
            pass 
        elif self.backend == 'adjacnet_matrix':
            pass  
        yield None 

    def bfs(self, src):
        assert isinstance(src, Vertex)
        if self.backend == 'VE':
            pass 
        elif self.backend == 'adjacent_list': 
            pass 
        elif self.backend == 'adjacnet_matrix':
            pass  
        yield None 


    # Do not modify this method

    @staticmethod
    def spring_layout(nodes, edges, iterations=50, k=0.1, repulsion=0.01):
        import numpy as np
        # Initialize positions randomly
        positions = {node: np.random.rand(2) for node in nodes}
        
        for _ in range(iterations):
            forces = {node: np.zeros(2) for node in nodes}
            
            # Repulsive forces between all pairs of nodes
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes):
                    if i != j:
                        diff = positions[node1] - positions[node2]
                        dist = np.linalg.norm(diff)
                        if dist > 0:
                            forces[node1] += (diff / dist) * repulsion / dist**2
            
            # Attractive forces for connected nodes
            for edge in edges:
                node1, node2 = edge.from_vertex, edge.to_vertex
                diff = positions[node2] - positions[node1]
                dist = np.linalg.norm(diff)
                
                if dist > 0:
                    force = k * (dist - 1)  # spring force
                    forces[node1] += force * (diff / dist)
                    forces[node2] -= force * (diff / dist)
            
            # Update positions
            for node in nodes:
                positions[node] += forces[node]
        
        return positions

    def show(self):
        import matplotlib.pyplot as plt
        
        if self.backend == 'VE':
            nodes = self.V 
            edges = self.E 
        elif self.backend == 'adjacent_list':
            nodes = self.adjacent_list.get_vertices()
            edges = self.adjacent_list.get_edges()
            
        positions = Graph.spring_layout(nodes, edges)
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # Plot edges
        for edge in edges:
            node1, node2 = edge.from_vertex, edge.to_vertex
            x_values = [positions[node1][0], positions[node2][0]]
            y_values = [positions[node1][1], positions[node2][1]]
            # ax.plot(x_values, y_values, color='gray', linewidth=2)
            plt.annotate('', xy = positions[edge.to_vertex], 
                             xytext = positions[edge.from_vertex], 
                             arrowprops={"facecolor": "red",  })


        # Plot nodes
        for node, pos in positions.items():
            ax.scatter(*pos, s=2000, color='lightblue')
            ax.text(*pos, node, fontsize=20, ha='center', va='center')

        ax.set_title("Graph Visualization with Spring Layout", fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()


if __name__ == '__main__':
    v1 = Vertex(0, 1)
    v2 = Vertex(1, 2)
    v3 = Vertex(2, 3)
    v4 = Vertex(3, 4)
    v5 = Vertex(4, 5)

    e1 = Edge(v1, v2) 
    e2 = Edge(v1, v3) 
    e3 = Edge(v2, v3)
    e4 = Edge(v2, v4)
    e5 = Edge(v3, v5) 
    e6 = Edge(v4, v5)

    V = [v1, v2]
    E = [e1]

    # g1 = Graph(V, E, backend='adjacent_list')
    g1 = Graph(V, E) 

    g1.add_vertex(v3)
    g1.add_vertex(v4)
    g1.add_vertex(v5)

    g1.add_edge(e2)
    g1.add_edge(e3)
    g1.add_edge(e4)
    g1.add_edge(e5)
    g1.add_edge(e6)
    
    # g1.remove_edge(e1)
    # g1.remove_edge(e2)
    # g1.remove_vertex(v3)


    # print(g1.get_vertices())
    # print(g1.adjacent_list.get_edges())
    # print(g1.get_neighbors(v1))
    # print(g1.adjacent_list.get_vertices())
    # edge_list = g1.adjacent_list.get_edges()
    # for x in edge_list:
    #     print(x.from_vertex, x.to_vertex)
    
    g1.show()



