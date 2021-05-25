# 베이스추정에 의한 정규분포추정
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 실제 분포
mu_true = 2.0
beta = 1.0

# 사전 분포
mu_0 = -2.0
beta_0 = 1.0

# 트레이닝 세트의 데이터 수
N_list = [2, 4, 10, 100]

# TODO 1열로 보여주는 것이 좋을 듯
fig1, axs = plt.subplots(1, len(N_list), figsize=(len(N_list) * 4, 4))
# fig1 = plt.figure()
# fig2 = plt.figure()

training_set = np.random.normal(loc=mu_true, scale=1.0 / beta, size=max(N_list))  # (size=(100,))


# TODO 함수명 바꾸기 (지난번 함수들과 공통점 찾아서 바꾸기)
def plot_fig1(c, t, mu_N, beta_N):
    # 평균 mu의 확률 분포
    var = 1.0 / beta_N
    linex = np.arange(-10, 10.1, 0.01)
    liney = norm.pdf(linex, loc=mu_N, scale=np.sqrt(var))  # pdf: probability density function

    # 트레이닝 셋 표시
    axs[c].set_title("N=%d" % N)
    axs[c].legend(loc=2)
    axs[c].set_xlim(-5, 5)
    axs[c].set_ylim(0)
    axs[c].plot(linex, liney, color='red', label="mu_N=%.2f\nvar=%.2f" % (mu_N, var))
    axs[c].scatter(t, [0.2] * N, marker='o', color='blue')


for idx, N in enumerate(N_list):
    # 트레이닝 셋 선택
    t = training_set[0:N]  # N개의 트레이닝 셋 (size=(N,1))

    # 사후 분포 계산
    mu_bar = np.mean(t)
    mu_N = (N * beta * mu_bar + beta_0 * mu_0) / (N * beta + beta_0)
    beta_N = N * beta + beta_0

    plot_fig1(idx, t, mu_N, beta_N)

# # 次に得られるデータの推定分布を表示
#     subplot = fig2.add_subplot(2,2,c+1)
#     subplot.set_title("N=%d" % n)
#     linex = np.arange(-10,10.1,0.01)
#
#     # 真の分布を表示
#     orig = norm(loc=mu_true, scale=np.sqrt(1.0/beta_true))
#     subplot.plot(linex, orig.pdf(linex), color='green', linestyle='--')
#
#     # 推定分布を表示
#     sigma = 1.0/beta_true+1.0/beta_N
#     mu_est = norm(loc=mu_N, scale=np.sqrt(sigma))
#     label = "mu_N=%.2f\nvar=%.2f" % (mu_N, sigma)
#     subplot.plot(linex, mu_est.pdf(linex), color='red', label=label)
#     subplot.legend(loc=2)
#
#     # トレーニングセットを表示
#     subplot.scatter(trainset, orig.pdf(trainset), marker='o', color='blue')
#     subplot.set_xlim(-5,5)
#     subplot.set_ylim(0)

fig1.suptitle("fig1")
fig1.tight_layout()
fig1.show()
# fig2.suptitle("fig2")
# fig2.tight_layout()
# fig2.show()
