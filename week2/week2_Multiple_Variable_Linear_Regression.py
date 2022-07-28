# 工作单位 ： 杭州电子科技大学
# 姓名 ： 罗会亮
# 开发时间 ： 2022/7/27 8:44

# 1.1的目标
# -扩展我们的回归模型例程以支持多种功能
# —扩展数据结构，支持多种特性
# 重写预测，成本和梯度例程，以支持多种功能
# -利用NumPy ' np。点来对它们的实现进行矢量化，以获得速度和简单性

# 1.2工具
# 在本实验室中，我们将利用:

# NumPy，一个用于科学计算的流行库
# Matplotlib，一个很流行的绘制数据的库

import copy,math
import numpy as np
import  matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  #控制输出结果的精度

# 你将使用房价预测的激励例子。训练数据集包含三个示例，其中四个特征(大小、卧室、楼层和年龄)如下表所示。
# 请注意，与早期实验室不同的是，尺寸是以平方英尺而不是1000平方英尺为单位的。
# 这将导致一个问题，你将在下一个实验室解决!
# 您将使用这些值建立一个线性回归模型，这样您就可以预测其他房子的价格。
# 例如，有1200平方英尺，3个卧室，1层，40年的房子。

# 您将使用这些值建立一个线性回归模型，这样您就可以预测其他房子的价格。
# 例如，有1200平方英尺，3个卧室，1层，40年的房子。
#
# 请运行以下代码单元格来创建X_train和y_train变量。

x_train = np.array([[2104,5,1,45],[1416,3,2,40],[852,2,1,35]])
y_train = np.array([460,232,178])

# print('x shape: {}, x type:{}'.format(x_train.shape,type(x_train)))
# # print(f'x shape: {x_train.shape}, x type:{type(x_train)}')
# print(x_train)
# print('y shape: {}, y type:{}'.format(y_train.shape,type(y_train)))
# print(y_train)

# 𝐰是一个包含𝑛个元素的向量。
# 每个元素包含与一个特性相关联的参数。
# 在我们的数据集中，n是4。
# 理论上，我们把它画成一个列向量

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
# print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

#点积函数
def predict_single_loop(x,w,b):
    '''

    :param x:
    :param w:
    :param b:
    :return:
    '''
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]
        p = p + p_i
    p = p + b
    return p

# x__vec = x_train[0,:] #获取x_train的第零行
# print('x__vec shape {}, x__vec value{}:'.format(x__vec.shape,x__vec))

# f_wb = predict_single_loop(x__vec,w_init,b_init)
# print('f_wb shape {}, prediction: {}'.format(f_wb.shape,f_wb))

#简单式点积函数
def predict(x,w,b):
    '''

    :param x:
    :param w:
    :param b:
    :return:
    '''

    p = np.dot(x,w) + b
    return p

x_vec = x_train[0,:]
# print('x_vec shape {}, x_vec value:{}'.format(x_vec.shape,x_vec))
f_wb = predict(x_vec,w_init,b_init)
# print('f_wb shape {}, prediction: {}'.format(f_wb.shape,f_wb))


def compute_cost(x,y,w,b):
    '''
    compute cost
    :param x:
    :param y:
    :param w:
    :param b:
    :return:
    '''
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x[i],w) + b
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost

cost = compute_cost(x_train,y_train,w_init,b_init)
# print('Cost at optional w: {}'.format(cost))


def compute_gradient(x,y,w,b):
    '''
    Computes the gradient for linear regression
    :param x:
    :param y:
    :param w:
    :param b:
    :return:
    '''
    m, n = x.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db,dj_dw

tmp_dj_db,tmp_dj_dw = compute_gradient(x_train,y_train,w_init,b_init)
# print('dj_db at initial w,b:{}'.format(tmp_dj_db))
# print('dj_dw at initial w,b: \n{}'.format(tmp_dj_dw))

def gradient_descent(x,y,w_in,b_in,cost_function,gradient_function,alpha,num_iters):
    '''
    执行批量梯度下降来学习通过取更新。
    学习速率为alpha的num_iter梯度步长
    :param x: Data, m examples with n features
    :param y: target values
    :param w_in: initial model parameters
    :param b_in: initial model parameter
    :param cost_function: function to compute cost
    :param gradient_function: function to compute the gradient
    :param alpha: Learning rate
    :param num_iters: 运行梯度下降的迭代次数
    :return:w,b
    '''

    # 一个数组用于存储每次迭代时的开销J和w，主要用于以后的绘图
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(x,y,w,b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000: #防止资源耗尽
            J_history.append(cost_function(x,y,w,b))

        # 每隔100次打印成本
        if i % math.ceil(num_iters / 10) == 0:
            #math.ceil 小数向上取整
            #math.floor()  “向下取整” ，即小数部分直接舍去
            #math.round()  “四舍五入”， 该函数返回的是一个四舍五入后的的整数
            print(f'Iteration {i:4d} : Cost {J_history[-1]:8.2f}   ')

    return w,b,J_history

initial_w = np.zeros_like(w_init) #.zeros_like()是指和括号里面的格式一下但是数值全为0
initial_b = 0.

iterations = 1000

alpha = 5.0e-7

w_final,b_final,J_hist =  gradient_descent(x_train,y_train,initial_w,initial_b,
                                         compute_cost,compute_gradient,alpha,iterations)


print(f'b,w found by gradient descent: {b_final:0.2f},{w_final}')
m,_ = x_train.shape
for i in range(m):
    print(f'prediction: {np.dot(x_train[i],w_final) + b_final:0.2f}, target value:{y_train[i]}')

# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step')
plt.show()