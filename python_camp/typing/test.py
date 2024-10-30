# from typing import Dict

def EvenOddChecker(number: dict[int, int]) -> str:
    if number[2] == 0:
        return 'even'
    else:
        return 'odd'

class A:
    c = 1
    @classmethod
    def f(A, a, b):
        print(type(a), type(b))
        print(A.c)

if __name__ == '__main__':
    # print(EvenOddChecker({5:5}))
    a = A()
    # a.f(1, 2)

    # A.f(a, 1, 2)
    A.f(1, 2)

