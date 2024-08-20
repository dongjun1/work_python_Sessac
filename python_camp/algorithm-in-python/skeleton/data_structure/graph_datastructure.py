class Vertex:
    def __init__(self, node_id, datum):
        self.datum = datum 
        self.node_id = node_id 

    def __eq__(self, other):
        if isinstance(self, Vertex) and isinstance(other, Vertex):
            return self.node_id == other.node_id
        return False 

    def __hash__(self):
        return hash((self.node_id, self.datum))

    def __str__(self):
        return str(self.datum)

# 방향 O : directed
# 방향 X : undirected
# edge 가 수치데이터를 가짐 : weighted
# 방향성이 사이클을 가짐 : cyclice <-> acyclic

class Edge:
    def __init__(self, from_vertex, to_vertex, is_directed = True, **data):
        assert isinstance(from_vertex, Vertex)    
        self.from_vertex = from_vertex

        assert isinstance(to_vertex, Vertex)
        self.to_vertex = to_vertex

        self.is_directed = is_directed
        self.data = data 
    
    def __eq__(self, other):
        if isinstance(self, Edge) and isinstance(other, Edge):
            return self.from_vertex == other.from_vertex and self.to_vertex == other.to_vertex

    def __str__(self):
        return f'{self.from_vertex} -> {self.to_vertex}'

class AdjList:
    def __init__(self, V, E):
        self.adjacent_list = {}
        for v in V:
            for e in E:
                if e.from_vertex == v:
                    self.adjacent_list[v] = [e]
                else :
                    self.adjacent_list[v] = []

    def add_vertex(self, v):
        self.adjacent_list[v] = []

    def remove_vertex(self, v):
        if v in self.adjacent_list.keys():
            self.adjacent_list.pop(v)

    def get_vertices(self):
        return self.adjacent_list.keys()
    
    def get_edges(self):
        edge_list = []
        for e in self.adjacent_list.values():
            for x in e:
                edge_list.append(x)
        return edge_list

    def remove_edge(self, e):
        for k in self.get_vertices():
            if e in self.adjacent_list[k]:
                for x in self.adjacent_list[k]:
                    del_val = []
                    del_val.append(x)
                    if e in del_val:
                        del_val.remove(e)
                    self.adjacent_list[k] = del_val
            
    
    def add_edge(self, e):
        for k in self.adjacent_list.keys():
            if k == e.from_vertex:
                self.adjacent_list[k] += [e]

    def get_neighbors(self, v):
        neighbor_vertex = []
        for data in self.adjacent_list[v]:
            neighbor_vertex.append(data.to_vertex.datum)
        return neighbor_vertex

class AdjMatrix:
    def __init__(self, V, E):
        pass 