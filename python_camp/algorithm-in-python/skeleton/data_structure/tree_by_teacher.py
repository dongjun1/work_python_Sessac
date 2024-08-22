try:
    from node import Node 
except ModuleNotFoundError:
    from data_structure.node import Node

class TreeNode:
    def __init__(self, node_id, datum):
        self.node_id = node_id
        self.datum = datum 

class Tree:
    def __init__(self, root, children = []):
        if not isinstance(root, TreeNode):
            root = TreeNode(root, root)
        self.root = root
        # children = children

        # for idx, child in enumerate(children):
        #     children[idx] = Tree(child)
        
        self.children = children

        

    def iter_nodes(self):
        yield self.root
        cur = self.children
        for node in cur:
            for n in node.iter_nodes():
                yield n
    
    def iter_nodes_with_address(self):
        yield [], self.root
        
        for i, node in enumerate(self.children):
            for addr, n in node.iter_nodes_with_address():
                yield [i] + addr, n
            

    def __iter__(self):
        yield self.root.datum
        cur = self.children
        for node in cur:
            for n in node:
                yield n
        
            
    def insert(self, address, elem):
        pass 

    def delete(self, address):
        addr_with_node = list(self.iter_nodes_with_address())
        del_value = 0
        for node in addr_with_node:
            if node[0] == address:
                del_value = node[-1].datum
        for node in self.iter_nodes():
            if del_value == node.datum:
                self.children.remove(node)
                
        return del_value

        
    def search(self, elem):
        addr_with_node = list(self.iter_nodes_with_address())
        for node in addr_with_node:
            if node[-1].datum == elem:
                return node[0]


    def root_datum(self):
        return self.root.datum
    
    def height(self):
        res = []
        for i in self.__iter__():
            res += [i]
        return len(str(res[-1]))
    

    def __str__(self):
        return '\n'.join(self.s())

    def s(t):
        res = [str(t.root.datum)]

        for child in t.children:
            part = child.s()
            for line in part:
                res.append('\t' + line)
        return res
        


if __name__ == '__main__':
    t1 = Tree(1, [
                Tree(11, [Tree(111), Tree(112)],), 
                Tree(12, [Tree(121), Tree(122), Tree(123),])
             ]
         )
    print(t1)
    
    # assert t1.root_datum() == 1 
    # assert t1.height() == 3
    # for node in t1.iter_nodes():
    #     print(node)
    # for addr, n in t1.iter_nodes_with_address():
    #     print(addr, n)
    # for addr, n in t1.iter_nodes_with_address():
    #     assert [int(e)-1 for e in list(str(n.datum))[1:]] == addr 
    #     assert t1.search(n.datum) == addr 

    # print(t1.delete([1, 1]))
    # print(t1)

    # t1.insert([2], Tree(13, [Tree(131), Tree(132), Tree(133)]))
    # t1.insert([1, 1], Tree(122, [Tree(1221), Tree(1222)]))

    # print(t1)
    
    # assert 122 == t1.delete([1,1])
    # assert 123 == t1.delete([1,2])

    # for addr, n in t1.iter_nodes_with_address():
    #     assert [int(e)-1 for e in list(str(n.datum))[1:]] == addr 
    #     assert t1.search(n.datum) == addr 

    # print(t1)