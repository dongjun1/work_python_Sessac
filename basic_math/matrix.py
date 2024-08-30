import numpy as np
import matplotlib.pyplot as plt

# create matrix 2x3
# a = np.array([
#     [2, 3, 4],
#     [10, 20, 30]
# ])

# b = np.array([
#     [2., 3., 4.],
#     [10., 20., 30.]
# ])
# # print(np.__version__)
# print(f'matrix a = \n{a}')
# print(f'shape of matrix a = {a.shape}')
# print(f'dtype of matrix a = {a.dtype}')

# print(f'matrix b = \n{b}')
# print(f'shape of matrix b = {b.shape}')
# print(f'dtype of matrix b = {b.dtype}')

# create diagonal matrix
# a = np.array([1, 2, 3, 4], dtype = np.float32)
# d1 = np.diag(a)
# print(f'matrix d1 = \n{d1}')
# print(f'shape of matrix d1 = {d1.shape}')
# print(f'dtype of matrix d1 = {d1.dtype}')

# b = np.random.randint(1, 10, 5)
# d2 = np.diag(b)
# print(f'matrix d2 = \n{d2}')
# print(f'shape of matrix d2 = {d2.shape}')
# print(f'dtype of matrix d2 = {d2.dtype}')

# create identity matrix
# i3 = np.eye(3)
# print(f'create 3x3 Identity \n {i3}')

# i5 = np.eye(5)
# print(f'create 5x5 Identity \n {i5}')

# create matrix transpose
# a = np.array([
#     [1, 2, 3, 4],
#     [5, 6, 7, 8]
# ], dtype = np.int64)

# print(f'matrix a :\n{a}')
# print(f'transpose a :\n{a.T}')

# matrix product
a = np.array([
    [1, 2],
    [3, 4]
])

b = np.array([
    [1, 1],
    [1, 2]
])

print(f'element product : \n{a * b}')
print(f'matrix product : \n{np.matmul(a, b)}')