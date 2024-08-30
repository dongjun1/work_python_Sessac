import numpy as np
import matplotlib.pyplot as plt

# v = np.array([1, 1])
# w = np.array([-2, -1])

# # directional field, vector : quiver()
# plt.quiver(0, 0, v[0], v[1], angles = 'xy', scale_units = 'xy', scale = 1, color = 'b')
# plt.quiver(0, 0, w[0], w[1], angles = 'xy', scale_units = 'xy', scale = 1, color = 'r')
# plt.grid(linestyle = ':')
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
# plt.show()

# v = np.array([
#     [1, 1], 
#     [-2, 2], 
#     [4, -7]
# ])

# origin = np.array([
#     [0, 0, 0],
#     [0, 0, 0]
# ])
# # print(*origin)
# # print(f'v = \n{v}')
# # print(f'shape of v = {v.shape}')
# # print(f'dtype of v = {v.dtype}')

# plt.quiver(*origin, v[:, 0], v[:, 1], angles = 'xy', scale_units = 'xy', scale = 1, color = ['r', 'b', 'c'])
# plt.grid(linestyle = ':')
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.show()

# projection vector
def projection_vector(base_vec, target_vec):
    scale = np.inner(base_vec, target_vec) / (np.linalg.norm(base_vec) ** 2)
    proj_vec = scale * base_vec
    plt.quiver(0, 0, base_vec[0], base_vec[1], angles = 'xy', scale_units = 'xy', scale = 1, color = 'r', label = 'base vector')
    plt.quiver(0, 0, target_vec[0], target_vec[1], angles = 'xy', scale_units = 'xy', scale = 1, color = 'g', label = 'target vector')
    plt.quiver(0, 0, proj_vec[0], proj_vec[1], angles = 'xy', scale_units = 'xy', scale = 1, color = 'b', label = 'projected vector')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(linestyle = ':')
    plt.legend()
    plt.show()

projection_vector(np.array([-2, 1]), np.array([-6, 4]))
