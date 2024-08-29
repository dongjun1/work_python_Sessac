import numpy as np
import matplotlib.pyplot as plt

# x = np.linspace(-4, 4, 100)
# y = 2 ** x
# y1 = 2 ** (x - 1)
# y2 = 2 ** (x - 1) + 1

# plt.plot(x, y, label = "2^x")
# plt.plot(x, y1, label = "2^(x-1)")
# plt.plot(x, y2, label = "2^(x-1) + 1", linestyle = ':')
# plt.grid(linestyle = ':')
# plt.vlines(0, 0, 16, colors = 'b')
# plt.legend()
# plt.show()


# x = np.linspace(-4, 4, 100)
# y = 2 ** -x
# y1 = 2 ** (-(x - 2))

# plt.plot(x, y, label = "2^-x")
# plt.plot(x, y1, label = "2^(-x+2)")
# plt.grid(linestyle = ':')
# plt.vlines(0, 0, 60, colors = 'b')
# plt.legend()
# plt.show()

# x = np.linspace(-3, 3, 100)
# # y = 2 ** x
# y_exp = np.exp(x)
# # y3 = 3 ** x
# y_exp2 = np.exp(-x)
# # plt.plot(x, y, label = "2^x")
# plt.plot(x, y_exp, label = "e^x")
# # plt.plot(x, y3, label = '3^x')
# plt.plot(x, y_exp2, label = "e^-x")
# plt.grid(linestyle = ':')
# plt.vlines(0, 0, 20, colors = 'b')
# plt.hlines(1, -3, 3, colors = 'g', linestyle = ':')
# plt.legend()
# plt.show()

# x = np.linspace(0, 4, 100)
# y = np.log2(x)
# y1 = np.log10(x)
# # y1 = -np.log2(x)
# y_exp = np.log(x)
# y_neg = np.emath.logn(0.5, x)

# plt.plot(x, y, label = 'log2(x)')
# plt.plot(x, y_neg, label = 'log0.5(x)')
# # plt.plot(x, y1, label = 'log10(x)')
# # plt.plot(x, y_exp, label = 'loge(x)')
# plt.grid(linestyle = ':')
# plt.hlines(0, 0, 4, color = 'violet', linestyle = ':')
# plt.legend()
# plt.show()

# Binary loss function
x = np.linspace(0, 4, 100)
y1 = -np.log(x)
y2 = -np.log(1 - x)

plt.plot(x, y1, label = '-log(x)')
plt.plot(x, y2, label = '-log(1 - x)')
plt.grid(linestyle = ':')
plt.vlines(1, -2, 3, color = 'violet', linestyle = ':')
plt.xlim(0, 1)
plt.legend()
plt.show()