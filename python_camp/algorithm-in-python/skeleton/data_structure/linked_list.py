try:
    from node import Node 
except ModuleNotFoundError:
    from data_structure.node import Node

class LinkedNode:
    def __init__(self, node_id, datum, next = None):
        self.node_id = node_id 
        self.datum = datum
        self.next = next 
        

class LinkedList:
    def __init__(self, elements):
        # len(int) -> int는 셀 수 없음.
        # if elements == [] or elements == ():
        if len(elements) == 0:
            self.head = None 
            self.tail = None 
            self.end = None
            self.size = 0
        else :
            if isinstance(elements, tuple) :
                elements = [x for x in elements]
            
            if len(elements[0]) == 2 :
                elements = self.sort_priority(elements)
                for i, e in enumerate(elements):
                    elements[i] = LinkedNode(i + 1, e)
                    
            else :
                for i, e in enumerate(elements) :
                    if not isinstance(i, LinkedNode) :
                        elements[i] = LinkedNode(len(elements)-i, e)
                
            for i in range(len(elements)) :
                i += 1
                if i == len(elements) :
                    break
                elements[-i].next = elements[-i-1]
                    
            self.head = elements[-1]
            self.tail = LinkedList(elements[-2:-len(elements)])
            self.end = elements[0]
            self.size = len(elements)
    
    def sort_priority(self, elem):
        n = len(elem)
        for i in range(n) :
            for j in range(n - i - 1) :
                if elem[j][-1] > elem[j+1][-1] :
                    elem[j], elem[j+1] = elem[j+1], elem[j]
        return elem

    def front(self):
        return self.head.datum
    
    def get_size(self):
        return self.size
    
    def is_empty(self):
        return self.size == 0


    def append(self, element):
        if not self.end == None:
            self.size += 1
            if not isinstance(element, LinkedNode):
                element = LinkedNode(self.end.node_id + 1, element)        
            self.tail.append(element)
            self.end.next = element
            self.end = element
     
        else :
            if not isinstance(element, LinkedNode):
                element = LinkedNode(1, element)
            self.head = element
            self.end = element
            self.tail = element
            self.size += 1
            

    def append_priority(self, element):
        # for i in self:
        #         if i[-1] == element.datum[-1] :
        #             element.next = i.next
        #             i.next = element
        pass
    
    def pop(self):
        pop_elem = self.head.datum
        self.head = self.head.next
        self.size -= 1
        return pop_elem
    
    def elements(self):
        return self.__iter__()

    def __iter__(self):
        cur = self.head
        while cur is not None:
            yield cur.datum
            cur = cur.next

    # def __iter__(self):
    #     res = []
    #     cur = self.head 
        
    #     while cur is not None:
    #         res.append(cur.datum)
    #         cur = cur.next
        
    #     res.reverse()
    #     return res
        

    def __str__(self):
        res = ''
        cur = self.head
        while cur is not None:
            res += str(cur.datum) + ',' + ' ' + str(cur.node_id) + ' '
            cur = cur.next

        return res.rstrip(',')

    

class DoublyLinkedNode(Node):
    def __init__(self, node_id, datum, prev = None, next = None):
        self.node_id = node_id 
        self.datum = datum
        self.next = next 
        self.prev = prev 

class DoublyLinkedList:
    def __init__(self, elements):
        if elements == []:
            self.head = None 
            self.tail = None 
            self.end = None
            self.size = 0
        else:
            self.head = None 
            self.tail = None 
            self.end = None
            self.size = 0

    def __iter__(self):
        yield None 

    def __str__(self):
        res = ''

        return res 

if __name__ == '__main__':
    # lst = LinkedList([1, 2, 3, 4])
    # print(lst.elements())
    # lst.append(5)
    # print(lst.elements())
    # lst.pop()
    # print(lst.elements())
    
    # q2 = LinkedList([])
    # q2.append(1)
    
    q2 = LinkedList([('c',1), ('d',4), ('e',2), ('b',3)])
    print(q2.elements() == [('c',1), ('e',2), ('b',3), ('d',4)])
    q2.append(('e', 3))
    print(q2)
    # assert lst.head.datum == 4
    # assert lst.head.next.datum == 3
    # assert lst.head.next.next.datum == 2
    # assert lst.head.next.next.next.datum == 1
    # assert lst.head.next.next.next.next is None
    # assert lst.head.next.next.next == lst.end
