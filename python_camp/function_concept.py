# --------------------------------------------
# 1. 함수의 다양한 입력들 살펴보기 
#
# 1) input이 없는 함수 
# 2) input이 여러 개 있는 함수 
# 3) input이 정해지지 않은 갯수만큼 있는 함수 
# --------------------------------------------

def pi():
    """원주율을 소숫점 두 자리까지 반환하는 함수
    """
    pi = '3.141592653589793238462643383279'
    return float(pi[:4])


def left_append(lst, elem):
    """lst의 왼쪽에 elem을 넣고, lst를 반환하는 함수 
    """
    new_lst = [elem] + lst
    return new_lst     

def left_extend(lst, *elem):
    """lst의 왼쪽에 정해지지 않은 갯수의 elem을 넣고 lst를 반환하는 함수 
    """
    elem_lst = [e for e in elem]
    new_lst = elem_lst + [l for l in lst]
    return new_lst

# --------------------------------------------
# 2. 함수의 call stack 알아보기 
# 
# 1) 아래 함수 b()를 실행할 때, 실행된 함수의 순서는?
# --------------------------------------------

def a():
    return pi()

# b()의 실행순서 : b() -> a() -> pi()
def b():
    return a()


# --------------------------------------------
# 2) 아래 함수 c()를 실행할 때, 실행된 함수의 순서와 각 함수의 input은? 
# --------------------------------------------

# c()의 실행순서 : c(lst) -> print(lst[0]) -> c(lst[1:])
# c(list(range(10)))에서 0~9까지의 리스트가 최초의 c함수의 input, 0출력 후 다시 c함수를 호출할 때 맨 앞의 값 하나를 빼고 다시 전달 따라서 input은 1~9. 0번째 인덱스의 값을 출력해야하니 
# 1을 출력 다시 1을 제외한 2~9까지의 리스트를 input으로 c함수를 호출.... 을 반복하여 마지막 값 까지 차례대로 출력.
def c(lst):
    print(lst[0])

    return c(lst[1:]) 

c(list(range(10)))
# print(pi())
# lst1 = [5, 654, 86, 8]
# lst2 = [45, 89, 58, 877, 636]
# print(left_append(lst, 6))

# print(left_extend(lst2, 65 ,88, 66))