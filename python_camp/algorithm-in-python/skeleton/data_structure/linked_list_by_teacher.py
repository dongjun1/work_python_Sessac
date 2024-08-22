try:
    from node import Node 
except ModuleNotFoundError:
    from data_structure.node import Node

class LinkedNode:
    def __init__(self, node_id, datum, next = None):
        self.node_id = node_id 
        self.datum = datum
        self.next = next 

    def set_next(self, next_node):
        assert isinstance(next_node, LinkedNode) or next_node is None

        self.next = next_node


class LinkedList:
    def __init__(self, elements):
    
        if not elements:
            self.head = None 
            self.tail = None 
            self.end = None
            self.size = 0
        else:
            elements = list(elements)

            for idx, elem in enumerate(elements):
                if not isinstance(elem, LinkedNode):
                   elements[idx] = LinkedNode(idx, elem)
                    
            self.head = elements[0]
            self.end = elements[-1]

            for idx, elem in enumerate(elements):
                if idx == len(elements)-1:
                    elem.next = None
                else:  
                    elem.next = elements[idx+1]
                
            self.tail = LinkedList(elements[1:])
            self.size = len(elements)

    

    def append_to_head(self, elem):
        if not isinstance(elem, LinkedNode):
            elem = LinkedNode(self.size, elem)
        elem.next = self.head
        self.head = elem

        if self.size == 0:
            self.end = elem
            self.tail = None
        
        self.size += 1

    def remove_from_head(self):
        res = self.head
        self.head = self.head.next
        self.size -= 1
        
        return res
    
    def append(self, elem):
        if not isinstance(elem, LinkedNode):
            elem = LinkedNode(self.size, elem)

        if self.end == None:
            self.head = elem

        self.end.next = elem
        self.end = elem
        self.size += 1
    
    def pop(self, idx):
        if self.size <= idx:
            raise IndexError('out of Index')

        cur = self.head
        cur_idx = 0

        while cur_idx < idx-1:
            cur = cur.next
            cur_idx += 1

        # cur가 idx-1번째 노드
        # idx-1      idx              idx+1
        # [cur]  -> [cur.next]  ->  [cur.next.next]

        # [cur] -------->>> [cur.next.next]
        if idx == 0:
            return self.remove_from_head()
        else:
            res = cur.next
            cur.next = cur.next.next
            self.size -= 1
            return res
    
    def insert(self, idx, elem):
        if not isinstance(elem, LinkedNode):
            elem = LinkedNode(self.size, elem)

        if self.size + 1 <= idx:
            raise IndexError('out of Index')

        cur = self.head
        cur_idx = 0

        while cur_idx < idx-1:
            cur = cur.next

        # cur가 idx-1번째 노드
        # idx-1      idx           idx+1
        # [cur]  -> [elem]  ->  [cur.next]
        if self.size == 0:
            self.head = elem
            self.end = elem
            elem.set_next(None)
        else:
            elem.set_next(cur.next)
            cur.set_next(elem)
        self.size += 1
        
    def __getitem__(self, idx):
        # lst[1]
        # LinkedList.__getitem__(1)

        if self.size <= idx:
            raise IndexError('out of Index')

        cur = self.head
        cur_idx = 0

        while cur_idx < idx:
            cur = cur.next
            cur_idx += 1
    
        return cur
    
    def __setitem__(self, idx, elem):
        # lst[1] = 3
        # LinkedList.__setitem__(1, 3)

        if self.size <= idx:
            raise IndexError('out of Index')

        cur = self.head
        cur_idx = 0

        while cur_idx < idx:
            cur = cur.next
            cur_idx += 1

        if isinstance(elem, LinkedNode):
            cur.datum = elem.datum
        else:
            cur.datum = elem


    def __iter__(self):
        current = self.head
        while current is not None:
            yield current.datum
            current = current.next
            
    def __str__(self):
        res = '[head] ->'

        for node in self:
            res += f'[{node}] ->'
        res += 'None'
        return res
        
        # current = self.head
        # while current is not None:
        #     res.append(str(current.datum))
        #     current = current.next
        # return ' -> '.join(res)

    def __len__(self):
        return self.size


class DoublyLinkedNode(Node):
    def __init__(self, node_id, datum, prev = None, next = None):
        self.node_id = node_id 
        self.datum = datum
        self.next = next 
        self.prev = prev 

class DoublyLinkedList:
    def __init__(self, elements):
        if elements is None:
            elements = []
        elements = list(elements)

        if not elements:
            self.head = None
            self.tail = None
            self.size = 0
       
        else:
            for idx, elem in enumerate(elements):
                if not isinstance(elem, DoublyLinkedNode):
                   elements[idx] = DoublyLinkedNode(idx, elem)
                    
            self.head = elements[0]
            self.end = elements[-1]

            for idx, elem in enumerate(elements):
                if idx == len(elements)-1:
                    elem.next = None
                else:  
                    elem.next = elements[idx+1]

            for idx, elem in enumerate(elements):
                if idx == 0:
                    elem.prev = None
                else:  
                    elem.prev = elements[idx-1]
                
            self.tail = DoublyLinkedList(elements[1:])
            self.size = len(elements)  

    def add_to_head(self, elem):
        new_node = DoublyLinkedNode(0, elem)
        
        if self.head is None:
            self.head = new_node
            self.end = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
            
        self.size += 1

    def delete_from_back(self):
        if self.size == 0:
            return None

        deleted_node = self.end

        if self.size == 1:
            self.head = None
            self.end = None
        else:
            self.end = self.end.prev
            self.end.next = None

        self.size -= 1
        return deleted_node.datum
        
    def __iter__(self):
        current = self.head
        while current is not None:
            yield current.datum
            current = current.next

    def __str__(self):
        res = []
        current = self.head
        while current is not None:
            res.append(str(current.datum))
            current = current.next
        return ' -> '.join(res) 

if __name__ == '__main__':
    lst = LinkedList([1,2,3])

    print(lst) 
    print(LinkedList.__str__(lst))
    str(lst)
    LinkedList.__str__(lst)
    