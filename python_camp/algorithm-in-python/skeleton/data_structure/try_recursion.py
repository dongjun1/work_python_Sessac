def move(lst):
    n = len(lst)
    A = lst
    B = []
    C = []

    
    # if n == 1
    if n == 1:
        A.remove(x for x in lst)
        C.append(lst)
        print(C)
    if n == 2:
        B.append(lst[0])
        A.remove(lst[0])
        C.append(lst[-1])
        C.append(lst[0])
        B.remove(lst[0])
        print(C)
        


move([1, 2])
    