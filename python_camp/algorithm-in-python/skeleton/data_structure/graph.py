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


class AdjList:
    def __init__(self, V, E):
        self.adjacent_list = {V : E}

    def add_vt(self, v):
        self.adjacent_list[v] = None

    def get_vts(self):
        return self.adjacent_list.keys()
    
    def add_eg(self, e):
        for k in self.adjacent_list.keys():
            if k == e.from_vertex:
                self.adjacent_list[k] = e

class AdjMatrix:
    def __init__(self, V, E):
        pass 