import random 
import os
# --------------------------------------------
# 1. max / min 구현하기 
#
# cmp라는 함수를 이용한 min/max 구현하기. 
# cmp는 두 원소 중 더 큰 것을 반환하는 함수. 
# --------------------------------------------
# def my_max_rule(lst) :
#     num = lst[0]
#     for i in lst :
#         if i > num :
#             num = i
#     return num

# def my_min_rule(lst) :
#     num = lst[0]
#     for i in lst[1:] :
#         if i < num :
#             num = i
#     return num

def my_max(lst, cmp = lambda x, y: x):
    num = lst[0]
    for elem in lst[1:] :
        num = cmp(num, elem)
    return num


def my_min(lst, cmp = lambda x, y: x):
    num = lst[0]
    for elem in lst[1:] :
        num = cmp(num, elem)
        if num > elem :
            num_min = elem
    return num_min


# print(my_max([500, 468, 687, 5555], cmp = lambda x, y: x if x >= y else y))
# print(my_min([500, 468, 687, 5555, 337], cmp = lambda x, y: x if x >= y else y))

# --------------------------------------------
# 2. sort 구현하기 
# 
# 1) 그냥 순서대로 오름차순으로 정렬하기 
# 2) 오름차순, 내림차순으로 정렬하기 
# 3) 주어진 기준 cmp에 맞춰서 오름차순, 내림차순으로 정렬하기 
# 4) 주어진 기준 cmp가 큰 element를 출력하거나, 같다는 결과를 출력하게 만들기 
# 5) cmp상 같은 경우 tie-breaking하는 함수 넣기 
# --------------------------------------------

def sort1(lst):
    n = len(lst)
    for i in range(n) :
        for j in range(n - i - 1) :
            if lst[j] > lst[j+1] :
                lst[j], lst[j+1] = lst[j+1], lst[j]

    return lst

def sort1_my_min(lst) :
    res = []
    copy = [e for e in lst]
    n = len(lst)

    while len(res) < n :
        m = my_min(copy, cmp = lambda x, y : x if x > y else y)
        res.append(m)
        lst.remove(m)
    
    return res

def get_insert_idx(res, elem) :
    idx = 0

    for i, e in enumerate(res) :
        if elem < e :
            return i
    
    return len(res)

def sort1_insert(lst) :
    res = []

    for idx, elem in enumerate(lst) :
        new_idx = get_insert_idx(res, elem)
        res.insert(new_idx, elem)
    return res

lst = [10, 1, 6, 4, 2, 5]    
# print(sort1_my_min(lst))
print(sort1_insert(lst))

def sort2(lst, upper_to_lower = True):
    if upper_to_lower  :
        return sort1(lst)
    else :
        reversed_lst = []
        sorted_lst = sort1(lst)
        n = len(lst)
        for i in range(1, n+1) :
            reversed_lst.append(sorted_lst[-i])
        return reversed_lst
    

def sort3(lst, upper_to_lower = True, cmp = lambda x, y: x):
    n = len(lst)
    if upper_to_lower :
        for i in range(n) :
            for j in range(n - i - 1) :
                if cmp(lst[j], lst[j+1]) == lst[j] :
                    lst[j], lst[j+1] = lst[j+1], lst[j]
    else :
        for i in range(n) :
            for j in range(n - i - 1) :
                if cmp(lst[j], lst[j+1]) == lst[j+1] :
                    lst[j], lst[j+1] = lst[j+1], lst[j]
    return lst
    # elif upper_to_lower == False :
    #     reversed_lst = []
    #     sorted_lst = sort1(lst)
    #     n = len(lst)
    #     for i in range(1, n+1) :
    #         reversed_lst.append(sorted_lst[-i])
    #     return reversed_lst 

def sort4(lst, upper_to_lower = True, cmp = lambda x, y: x):
    pass 


def sort5(lst, upper_to_lower = True, cmp = lambda x, y: x, tie_breaker = lambda x, y: random.choice([x,y])):
    n = len(lst)
    if upper_to_lower :
        for i in range(n) :
            for j in range(n - i - 1) :
                if cmp(lst[j], lst[j+1]) == 'same' :
                    tie = tie_breaker(lst[j], lst[j+1])
                    if tie == lst[j] :
                        lst[j], lst[j+1] = tie, lst[j+1]
                    else :
                        lst[j], lst[j+1] = lst[j], tie
                elif cmp(lst[j], lst[j+1]) == lst[j] :
                    lst[j], lst[j+1] = lst[j+1], lst[j]
    else :
        for i in range(n) :
            for j in range(n - i - 1) :
                if cmp(lst[j], lst[j+1]) == 'same' :
                    tie = tie_breaker(lst[j], lst[j+1])
                    if tie == lst[j] :
                        lst[j], lst[j+1] = tie, lst[j+1]
                    else :
                        lst[j], lst[j+1] = lst[j], tie
                elif cmp(lst[j], lst[j+1]) == lst[j+1] :
                    lst[j], lst[j+1] = lst[j+1], lst[j]
    return lst


# 주어진 기준 cmp가 큰 element를 출력하거나, 같다는 결과를 출력하게 만들기 
def compare(x, y) :
    if x > y :
        return x 
    elif x < y :
        return y
    elif x == y :
        return "same"

rand_lst = [random.randint(0, 100) for i in range(50)]
# print(rand_lst) 
# print(sort1(rand_lst))
# print(sort1(rand_lst) == sorted(rand_lst))
# print(sort1([10, 6, 7, 4, 8, 2, 9, 1]))
# print(sort2(rand_lst))
# print(list(reversed(sorted(rand_lst))))
# print(sort2(rand_lst, upper_to_lower = False) == list(reversed(sorted(rand_lst))))
# print(sort2(rand_lst, upper_to_lower = False))
# print(sort3(rand_lst, cmp = lambda x, y : x if x > y else y) == sorted(rand_lst))
# print(sort3(rand_lst, upper_to_lower = False, cmp = lambda x, y : x if x > y else y) == list(reversed(sorted(rand_lst))))
# print(sort5(rand_lst, upper_to_lower = False, cmp = compare) == list(reversed(sorted(rand_lst))))
# print(sort5(rand_lst, upper_to_lower = True, cmp = compare) == sorted(rand_lst))
lt = [(1, '하트1'), (3, 4), (98, 0), (1, '스페이드2'), (1, '하트10')]
# print(sort5(lt, upper_to_lower = False, cmp = compare) == list(reversed(sorted(lt))))
# print(sort5(lt, upper_to_lower = True, cmp = compare) == sorted(lt))
# print(sort5(lt, upper_to_lower = True, cmp = compare) == sorted(lt))
# print(sort5(lt, upper_to_lower = False, cmp = compare))
# print(sort5(lt, upper_to_lower = False, cmp = compare) == list(reversed(sorted(lt))))



# --------------------------------------------
# os_file_concept.py 해보고 올 것 
# --------------------------------------------

# --------------------------------------------
# 3. safe pickle load/dump 만들기 
# 
# 일반적으로 pickle.load를 하면 무조건 파일을 읽어와야 하고, dump는 써야하는데 반대로 하면 굉장히 피곤해진다. 
# 이런 부분에서 pickle.load와 pickle.dump를 대체하는 함수 safe_load, safe_dump를 짜 볼 것.  
# hint. 말만 어렵고 문제 자체는 정말 쉬운 함수임.
# --------------------------------------------
import pickle

def safe_load(pickle_path):
   return pickle.load(open(pickle_path, 'rb'))

def safe_dump(pickle_path):
    return pickle_path(open(pickle_path, 'wb+'))


# --------------------------------------------
# 4. 합성함수 (추후 decorator)
# 
# 1) 만약 result.txt 파일이 없다면, 함수의 리턴값을 result.txt 파일에 출력하고, 만약 있다면 파일 내용을 읽어서 
#    '함수를 실행하지 않고' 리턴하게 하는 함수 cache_to_txt를 만들 것. txt 파일은 pickle_cache 폴더 밑에 만들 것.  
# 2) 함수의 실행값을 input에 따라 pickle에 저장하고, 있다면 pickle.load를 통해 읽어오고 없다면 
#    실행 후 pickle.dump를 통해 저장하게 하는 함수 cache_to_pickle을 만들 것. pickle 파일은 pickle_cache 폴더 밑에 만들 것. 
# 3) 함수의 실행값을 함수의 이름과 input에 따라 pickle에 저장하고, 2)와 비슷하게 진행할 것. pickle 파일은 pickle_cache 폴더 밑에, 각 함수의 이름을 따서 만들 것 
# --------------------------------------------

def check_dir_file(dir_name, file_name) :
    return f"create {dir_name} and {file_name}"
        
        

def cache_to_txt(function):
    present_dir = os.getcwd()
    dir_name = 'pickle_cache'
    file_name = 'result.txt'
    if not os.path.exists(dir_name) :
        os.makedirs(dir_name)
        print(f'{dir_name} does not exists;')
    elif os.path.exists(dir_name) :
        if file_name not in os.listdir() :
            print(132)
            f = open(present_dir + '\\' + dir_name + '\\' + file_name, 'w+', encoding = 'utf-8')
            return print(function(dir_name, file_name), file = f)            
        else :
            r = open(present_dir + '\\' + dir_name + '\\' + file_name, 'r', encoding = 'utf-8')  
            return r
    else :
        return f'{dir_name} does exists;'


def cache_to_pickle(function):
    pass
     


# cache_to_txt(check_dir_file)