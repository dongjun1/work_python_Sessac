import numpy as np
import matplotlib.pyplot as plt

# 다항함수
# def polynomial(x, n): 
#     y = x**n
#     return y

# x = np.linspace(-4, 4, 100)
# # print(x)
# y = polynomial(x, 1) # Linear Function
# y2 = polynomial(x, 2)
# y3 = polynomial(x, 3)

# plt.plot(x, y, label = "Y = X")
# plt.plot(x, y2, label = "Y = X^2")
# plt.plot(x, y3, label = "Y = X^3")
# plt.plot(x, polynomial(x, 4), label = "Y = X^4")

# plt.title("Polynomial Curve")
# plt.xlabel("X")
# plt.ylabel("f(x)")
# plt.legend()
# plt.ylim(-2, 2)
# plt.xlim(-2, 2)

# plt.grid(linestyle = ":")
# plt.show()

# 삼각함수
# -2pi ~ 2pi 까지의 Radian 값 생성.
# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# # print(x)
# sin_x = np.sin(x)
# cos_x = np.cos(x)
# tan_x = np.tan(x)
# plt.figure(figsize = (12, 5))
# # plt.plot(x, sin_x, label = "sin(x)")
# # plt.plot(x, cos_x, label = "cos(x)")
# plt.plot(x, tan_x, label = "tan(x)")
# plt.vlines(0, -1, 1, colors = 'red')
# plt.hlines(0, -2*np.pi, 2*np.pi, colors = 'black')
# plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], labels = ["-2pi", "-pi", "0", "pi", "2pi"])

# plt.ylim(-1, 1)
# plt.legend()
# plt.grid(linestyle = ":")

# plt.show()

# sin, cos, tan 함수를 각각 한번에 출력하기
# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# sin_x = np.sin(x)
# cos_x = np.cos(x)
# tan_x = np.tan(x)

# y_list = [sin_x, cos_x, tan_x]
# y_label = ['sin(x)', 'cos(x)', 'tan(x)']
# color_list = ["b", "g", "r"]
# fig, axis = plt.subplots(3, 1, figsize = (12, 10))

# for i in range(3):
#     axis[i].plot(x, y_list[i], label = y_label[i], color = color_list[i])
#     axis[i].set_ylim(-1, 1)
#     axis[i].grid(linestyle = ":")
#     axis[i].set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], labels = ["-2pi", "-pi", "0", "pi", "2pi"])

# plt.show()

# (1) y - b = sin(x - a)의 그래프는 y = sin(x)의 그래프를 x축으로 a, y축으로 b만큼 평행 이동한 것이다.
# (1) y = sin(x - a) + b의 그래프는 y = sin(x)의 그래프를 x축으로 a, y축으로 b만큼 평행 이동한 것이다.

# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# y = np.sin(x)
# # x - pi : x축을 pi만큼 양의 방향으로 이동. + 1 : y축을 1만큼 양의 방향으로 이동.
# y1 = np.sin(x - np.pi) + 1

# plt.figure(figsize = (12, 5))
# plt.plot(x, y, label = "sin(x)")
# plt.plot(x, y1, label = "sin(x - pi) + 1")
# plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], labels = ["-2pi", "-pi", "0", "pi", "2pi"])
# plt.grid(linestyle = ':')
# plt.legend()
# plt.show()

# (2) y = -sin(x)의 그래프는 y = sin(x)의 그래프에 대해 x축을 중심으로 선대칭이다.

# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# y = np.sin(x)

# y1 = -np.sin(x)

# plt.figure(figsize = (12, 5))
# plt.plot(x, y, label = "sin(x)")
# plt.plot(x, y1, label = "-sin(x)", linestyle = ':')
# plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], labels = ["-2pi", "-pi", "0", "pi", "2pi"])
# plt.grid(linestyle = ':')
# plt.legend()
# plt.show()

# (2) y = -cos(x)의 그래프는 y = cos(x)의 그래프에 대해 x축을 중심으로 선대칭이다.

# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# y = np.cos(x)

# y1 = -np.cos(x)

# plt.figure(figsize = (12, 5))
# plt.plot(x, y, label = "cos(x)")
# plt.plot(x, y1, label = "-cos(x)", linestyle = ':')
# plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], labels = ["-2pi", "-pi", "0", "pi", "2pi"])
# plt.grid(linestyle = ':')
# plt.legend()
# plt.show()

# (3) y = a * sin(x)의 그래프는 y = sin(x)의 a배이다.

# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# y = np.sin(x)

# y1 = 2 * np.sin(x)

# plt.figure(figsize = (12, 5))
# plt.plot(x, y, label = "sin(x)")
# plt.plot(x, y1, label = "a * sin(x)", linestyle = ':')
# plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], labels = ["-2pi", "-pi", "0", "pi", "2pi"])
# plt.grid(linestyle = ':')
# plt.legend()
# plt.show()

# (4) y = sin(ax)의 그래프는 y = sin(x)와 모양은 같고 주기가 2pi / |a|인 그래프이다.

# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# y = np.sin(x)
# y1 = np.sin(2 * x)

# plt.figure(figsize = (12, 5))
# plt.plot(x, y, label = "sin(x)")
# plt.plot(x, y1, label = "sin(2x)", linestyle = ':')
# plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], labels = ["-2pi", "-pi", "0", "pi", "2pi"])
# plt.grid(linestyle = ':')
# plt.legend()
# plt.show()

# y = sin(x - (np.pi / 2))와 y = sin(x - (np.pi / 2)) + 1의 그래프 그려보기.
# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# y = np.sin(x - (np.pi / 2))
# y1 = np.sin(x - (np.pi / 2)) + 1

# plt.figure(figsize = (12, 5))
# plt.plot(x, y, label = "sin(x - (np.pi / 2))")
# plt.plot(x, y1, label = "sin(x - (np.pi / 2)) + 1", linestyle = ':')
# plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], labels = ["-2pi", "-pi", "0", "pi", "2pi"])
# plt.grid(linestyle = ':')
# plt.legend()
# plt.show()

# y = 2 * sin(x)와 y = 2 * sin(x - (np.pi / 2)) + 1의 그래프 그려보기.
# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# y = 2 * np.sin(x)
# y1 = 2 * np.sin(x - (np.pi / 2)) + 1

# plt.figure(figsize = (12, 5))
# plt.plot(x, y, label = "2sin(x)")
# plt.plot(x, y1, label = "2sin(x - (np.pi / 2)) + 1", linestyle = ':')
# plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], labels = ["-2pi", "-pi", "0", "pi", "2pi"])
# plt.grid(linestyle = ':')
# plt.legend()
# plt.show()

# y = sin(2x)와 y = sin(2(x - (np.pi / 2))) - 1의 그래프 그려보기.
# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# y = np.sin(2 * x)
# y1 = np.sin(2 * (x - (np.pi / 2))) - 1

# plt.figure(figsize = (12, 5))
# plt.plot(x, y, label = "sin(2x)")
# plt.plot(x, y1, label = "sin(2 * (x - (np.pi / 2))) - 1", linestyle = ':')
# plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], labels = ["-2pi", "-pi", "0", "pi", "2pi"])
# plt.grid(linestyle = ':')
# plt.legend()
# plt.show()

# (5) y = cos(-x)의 그래프와 y = cos(x)의 그래프 그려보기.
# 동일한 그래프가 찍혀 나오는 것은 cos함수는 우함수이기 때문.

# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# y = np.cos(x)
# y1 = np.cos(-x)

# plt.figure(figsize = (12, 5))
# plt.plot(x, y, label = "cos(x)")
# plt.plot(x, y1, label = "cos(-x)", linestyle = ':')
# plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], labels = ["-2pi", "-pi", "0", "pi", "2pi"])
# plt.grid(linestyle = ':')
# plt.legend()
# plt.show()


# (6) y = sin(-x)의 그래프와 y = sin(x)의 그래프 그려보기.
# sin함수는 기함수이기 때문에 대칭으로 출력.

# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# y = np.sin(x)
# y1 = np.sin(-x)

# plt.figure(figsize = (12, 5))
# plt.plot(x, y, label = "sin(x)")
# plt.plot(x, y1, label = "sin(-x)", linestyle = ':')
# plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], labels = ["-2pi", "-pi", "0", "pi", "2pi"])
# plt.grid(linestyle = ':')
# plt.legend()
# plt.show()

# y = cos(x - pi / 2)의 그래프

# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# y = np.cos(x - (np.pi / 2))

# plt.figure(figsize = (12, 5))
# plt.plot(x, y, label = "cos(x - (np.pi / 2))")
# plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], labels = ["-2pi", "-pi", "0", "pi", "2pi"])
# plt.grid(linestyle = ':')
# plt.legend()
# plt.show()

# y = -cos(x) + 1의 그래프

# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# y = -np.cos(x) + 1

# plt.figure(figsize = (12, 5))
# plt.plot(x, y, label = "cos(x - (np.pi / 2))")
# plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], labels = ["-2pi", "-pi", "0", "pi", "2pi"])
# plt.grid(linestyle = ':')
# plt.legend()
# plt.show()

# y = -cos(2x + pi)의 그래프

# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# y = np.cos(2 * x + np.pi)

# plt.figure(figsize = (12, 5))
# plt.plot(x, y, label = "cos(2 * x + np.pi)")
# plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], labels = ["-2pi", "-pi", "0", "pi", "2pi"])
# plt.grid(linestyle = ':')
# plt.legend()
# plt.show()

# cos
x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = 2 * np.cos(2 * x) - 1

plt.figure(figsize = (12, 5))
plt.plot(x, np.cos(x), label = "cos(x)")
plt.plot(x, y, label = "2 * cos(2 * x) - 1")
plt.legend()

plt.grid(linestyle = ':')
plt.show()