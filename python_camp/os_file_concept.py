import os 
import pickle 
from time import time
# --------------------------------------------
# 1. os 기초 예제 
# 
# 1) os.path 이해하기 (os.path.exists, os.path.join, os.path)
# 2) os.listdir / os.makedir 해보기 
# 3) os.getcwd / os.changedir 해보기 
# --------------------------------------------

# cwd = current working dir
# print(os.getcwd())
# print(os.path.getatime(os.getcwd()))
# print(os.path.getmtime(os.getcwd()))
# for elem in os.listdir() :
#     # print(elem)
    
#     if os.path.isdir(elem) :
#         print('<DIR>\t\t' + elem)
#     elif '.' in elem :
#         extension = elem.split('.')[-1]
#         print(f'{extension} file\t\t{elem}')

# def create_dir(dir_name) :
    
#     if not os.path.exists(dir_name) :
#         print(f'{dir_name} does not exists;')
#         os.makedirs(dir_name)
    
#     else :
#         print(f'{dir_name} does exists;')

# create_dir('hello_world')


# --------------------------------------------
# 2. file 기초 예제 
# 
# 1) open 이해하기 
# 2) 파일 읽기, 써보기 
# --------------------------------------------

# a : append
# w : write
# r : read
# w+ : if file is exist : write, not exists create
# f.read() : 파일을 한번에 읽어옴
# f.readline() : 파일을 한 줄씩 읽어옴
# f.readlines() : 파일을 한 줄씩 끝까지 읽어옴

# begin = time()
# f = open('example.txt', 'w+', encoding = 'utf-8')

# for i in range(1000000) :
#     print(str(i) * 100, file = f)

# f.close()

# end = time()
# print(f'{end - begin} seconds')

# # f = open('example.txt', 'r', encoding = 'utf-8')
# # print(f.readline())
# # print(f.readline())
# # for line in f.readlines():
# #     print(line)
    
# f.close()


# --------------------------------------------
# 3. pickle 기초 예제 
# 
# 1) pickle.load() 해보기 
# 2) pickle.dump() 해보기 
# --------------------------------------------

# b : byte
# lambda 식 같은 경우는 pickle 불가능.
d = {'A' : 98, 'B' : 60}

pickle.dump(d, open('empty_dict.pickle', 'wb+'))

e = pickle.load(open('empty_dict.pickle', 'rb'))

print(e)