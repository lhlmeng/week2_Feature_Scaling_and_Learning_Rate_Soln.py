# 工作单位 ： 杭州电子科技大学
# 姓名 ： 罗会亮
# 开发时间 ： 2022/7/25 8:10

import math,copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x,plt_contour_wgrad,plt_divergence,plt_gradients

x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])

# # 𝑓𝑤,𝑏(𝑥(𝑖))=𝑤𝑥(𝑖)+𝑏                                  公式(1)
# 𝐽(𝑤,𝑏)=12𝑚∑𝑖=0𝑚−1(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))**2                 公式(2)
# w = w - alpha * dj_dw
# b = b - alpha * dj_db                                公式(3)
#
# dj_dw = (1 / m) * ∑𝑖=0𝑚−1(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖)) * 𝑥(𝑖)      公式(4)
# dj_db = (1 / m) * ∑𝑖=0𝑚−1(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))             公式(5)


#Function to calculate the cost
def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb * x[i] - y[i]) ** 2
        pass
    total_cost =1 / (2 * m) * cost

    return total_cost

def compute_gradient(x,y,w,b):
    '''
    Compute the gradient for linear regression
    :param x:
    :param y:
    :param w:
    :param b:
    :return:
    dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
    dj_db (scalar): The gradient of the cost w.r.t. the parameter b
    '''

    #Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw,dj_db

plt_gradients(x_train,y_train,compute_cost,compute_gradient)
plt.show()

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    '''
    对w,b进行梯度下降拟合。通过获取更新w,b。
    学习速率为alpha的num_iter梯度步长
    :param x: m examples
    :param y: target values
    :param w_in:模型参数的初始值
    :param b_in:模型参数的初始值
    :param alpha:学习率
    :param num_iters:运行梯度下降的迭代次数
    :param cost_function:函数调用产生成本
    :param gradient_function:函数来调用以产生梯度
    :return:
    w (scalar): 运行梯度下降后更新参数值
    b (scalar): 运行梯度下降后更新参数值
    J_history (List): 历史成本值
    p_history (list): 参数[w,b]的历史
    '''

    #防止修改全局变量w_in
    w = copy.deepcopy(w_in)
    #copy.deepcopy()函数是一个深复制函数。
    # 所谓深复制，就是从输入变量完全复刻一个相同的变量，无论怎么改变新变量，原有变量的值都不会受到影响。

    #
    #在每次迭代时存储成本J和w的数组，主要用于以后的绘图
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # 计算梯度并使用gradient_function更新参数
        dj_dw, dj_db = gradient_function(x,y,w,b)

        #使用公式(3)更新参数
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    # 保存每次迭代成本J
    if i < 100000:  #防止资源耗尽
        J_history.append(cost_function(x,y,w,b))
        p_history.append([w, b])

    # 如果 < 10，则每隔10次或相同次数打印成本
    if i % math.ceil(num_iters / 10) == 0:
        print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
              f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
              f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w,b,J_history,p_history

# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()

print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")


fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)


fig, ax = plt.subplots(1,1, figsize=(12, 4))
plt_contour_wgrad(x_train, y_train, p_hist, ax, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5],
            contours=[1,5,10,20],resolution=0.5)


# initialize parameters
w_init = 0
b_init = 0
# set alpha to a large value
iterations = 10
tmp_alpha = 8.0e-1
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)

# plt_divergence(p_hist, J_hist,x_train, y_train)
plt.show()