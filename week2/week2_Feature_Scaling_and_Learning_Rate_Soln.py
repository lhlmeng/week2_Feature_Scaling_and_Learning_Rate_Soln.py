# 工作单位 ： 杭州电子科技大学
# 姓名 ： 罗会亮
# 开发时间 ： 2022/7/27 20:04

import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0';
plt.style.use('./deeplearning.mplstyle')
from lab_utils_multi import  load_house_data, compute_cost, run_gradient_descent
from lab_utils_multi import  norm_plot, plt_contour_multi, plt_equal_scale, plot_cost_i_w

x_train,y_train = load_house_data()
x_features = ['size(sqft)','bedrooms','floors','age']

fig,ax = plt.subplots(1,4,figsize=(12,3),sharey=True) #1,4是表示有一行四个表格
#sharey=True 表示y轴共享一个刻度

for i in range(len(ax)):
    ax[i].scatter(x_train[:,i],y_train)
    ax[i].set_xlabel(x_features[i])
ax[0].set_ylabel("Price (1000's)")
# plt.show()

# _,_,hist = run_gradient_descent(x_train,y_train,10,alpha = 9.9e-7)

# plot_cost_i_w(x_train, y_train, hist)

#set alpha to 1e-7
# _,_,hist = run_gradient_descent(x_train, y_train, 10, alpha = 1e-7)
# plot_cost_i_w(x_train,y_train,hist)


def zscore_normalize_features(x):
    '''

    Args:
        X:

    Returns:

    '''
    mu = np.mean(x,axis = 0)
    sigma = np.std(x,axis = 0) #求标准差
    x_norm = (x - mu) / sigma

    return x_norm,mu,sigma

mu     = np.mean(x_train,axis=0)
sigma  = np.std(x_train,axis=0)
x_mean = (x_train - mu)
x_norm = (x_train - mu)/sigma

fig,ax=plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(x_train[:,0], x_train[:,3])
ax[0].set_xlabel(x_features[0]); ax[0].set_ylabel(x_features[3]);
ax[0].set_title("unnormalized")
ax[0].axis('equal')

ax[1].scatter(x_mean[:,0], x_mean[:,3])
ax[1].set_xlabel(x_features[0]); ax[0].set_ylabel(x_features[3]);
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(x_norm[:,0], x_norm[:,3])
ax[2].set_xlabel(x_features[0]); ax[0].set_ylabel(x_features[3]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
# plt.show()

# normalize the original features
x_norm, x_mu, x_sigma = zscore_normalize_features(x_train)
print(f"X_mu = {x_mu}, \nX_sigma = {x_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(x_train,axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(x_norm,axis=0)}")

fig,ax=plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i],x_train[:,i],)
    ax[i].set_xlabel(x_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features before normalization")
# plt.show()
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],x_norm[:,i],)
    ax[i].set_xlabel(x_features[i])
ax[0].set_ylabel("count");
fig.suptitle(f"distribution of features after normalization")

# plt.show()

w_norm, b_norm, hist = run_gradient_descent(x_norm, y_train, 1000, 1.0e-1, )

#predict target using normalized features
m = x_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(x_norm[i], w_norm) + b_norm

    # plot predictions and targets versus original features
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(x_features[i])
    ax[i].scatter(x_train[:,i],yp,color=dlorange, label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

# First, normalize out example.
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - x_mu) / x_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")

plt_equal_scale(x_train, x_norm, y_train)