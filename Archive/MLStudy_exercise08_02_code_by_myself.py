#%%
# 베이스 추정에 의한 회귀 분석
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
#파라미터 설정
beta_true = 1.0 / (0.3) ** 2
alpha = 1.0 / 100 ** 2
M = 9  # 다항식 차수
N_list = [4, 5, 10, 100]
beta = beta_true

#%%
# 트레이닝 셋 {x_n,y_n} (n=1...N) 을 준비
def generate_training_set(num):
    dataset = pd.DataFrame(columns=["x", "y"])
    for idx in range(num):
        x = float(idx) / float(num - 1)
        y = np.sin(2.0 * np.pi * x) + np.random.normal(scale=0.3)
        dataset = dataset.append(pd.Series([x, y], index=["x", "y"]), ignore_index=True)
    return dataset

# 평균 m(x)
# phi.T: (size=(1,M+1))
# S: (size=(M+1,M+1))
# sigma_t_phi: (size=(M+1,1))
def mean_fun(x):
    t = training_set.y
    phi_x = np.array([x ** i for i in range(0, m + 1)])
    sigma_t_phi= np.sum(np.multiply(t.values[:,np.newaxis],phis.values),axis = 0)
    #TODO flatten지웠는데 괜찮을까?
    # m_x = np.linalg.multi_dot([beta*phi_x.T, s, sum]).flatten()
    m_x = np.linalg.multi_dot([beta*phi_x.T, s, sigma_t_phi]).flatten()
    return m_x

# TODO: 분산이 계산된거니 표준편차로 바꿔서 대입시키기
def deviation_fun(x):
    phi_x = pd.DataFrame([x ** i for i in range(0, m + 1)])
    variance = 1.0 / beta + np.linalg.multi_dot([phi_x.T, s, phi_x])
    # TODO diagonal()제거 했는데 괜찮을지 확인하기
    # return variance.diagonal()
    return np.sqrt(variance).diagonal()

def f(x,ws):
    # x : (size=(n,1))
    # ws : (size=(M+1))
    x_n = np.array([x ** i for i in range(0, m + 1)])
    return np.multiply(ws[:,np.newaxis],x_n).sum(axis = 0)

#%%
m = 9
N = 5
idx = 0
training_set = generate_training_set(N)
t = training_set.y
c = 0

phis = pd.DataFrame() # (size=(N,M+1))
for i in range(0, m + 1):
    p = training_set.x ** i
    p.name = "x**%d" % i
    phis = pd.concat([phis, p], axis=1)

# 분산(S) 계산
phiphi_sum = 0

for _, phi in phis.iterrows():
    phi = phi.values[:,np.newaxis] # reshape the phi (M+1,) to (M+1,1)
    phiphi_sum += np.dot(phi,phi.T) # (size=(M+1,M+1))
s_inv = alpha * pd.DataFrame(np.identity(m + 1)) + beta * phiphi_sum
s = np.linalg.inv(s_inv)  # 사후분포의 공분산행렬

sigma_t_phi= np.sum(np.multiply(t.values[:,np.newaxis],phis.values),axis = 0)
mean = np.linalg.multi_dot([beta*s,sigma_t_phi])

# Main
fig, axs = plt.subplots(len(N_list),2, figsize=(8,len(N_list)*4))
c = 0

df_ws = pd.DataFrame()
ws_samples = pd.DataFrame(np.random.multivariate_normal(mean,s,4))

# 그래프 이름 및 범위 설정
axs[c][0].set_xlim(-0.05,1.05)
axs[c][0].set_ylim(-2,2)
axs[c][0].set_title("N=%d" % N)
axs[c][1].set_xlim(-0.05,1.05)
axs[c][1].set_ylim(-2,2)
axs[c][1].set_title("N=%d" % N)

# 트레이닝 셋 표시
axs[c][0].scatter(training_set.x, training_set.y, marker='o', color='blue')
axs[c][1].scatter(training_set.x, training_set.y, marker='o', color='blue')

line_x = np.arange(0, 1.01, 0.01)

# 실제 값 계산
line_y_true = np.sin(2 * np.pi * line_x)

# 평균과 표준편차 곡선 계산
line_y_mu = np.array(mean_fun(line_x))
line_y_d = np.array(deviation_fun(line_x))

# 실제값 표시
axs[c][0].plot(line_x, line_y_true, color='green', linestyle=':')

# 평균과 표준편차 곡선 표시
axs[c][0].plot(line_x, line_y_mu, color='red', label='mean')
axs[c][0].legend(loc=1)
axs[c][0].plot(line_x, line_y_mu - line_y_d, color='black', linestyle='--')
axs[c][0].plot(line_x, line_y_mu + line_y_d, color='black', linestyle='--')

# 평균 곡선 표시
axs[c][1].plot(line_x, line_y_mu, color='red', label="mean")
axs[c][1].legend(loc=1)

for index, ws in ws_samples.iterrows():
    line_y_f = f(line_x, ws)
    axs[c][1].plot(line_x, line_y_f, color='green', linestyle='--')
fig.show()