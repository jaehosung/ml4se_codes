import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import randint, rand
import urllib.request
# # 이미지 불러오기
# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/jaehosung/ml4se_codes/main/exercise07/sample-images_01.txt", "sample-images_01.txt"
# )
# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/jaehosung/ml4se_codes/main/exercise07/sample-labels_01.txt", "sample-labels_01.txt"
# )

# 트레이닝 세트 읽기
df = pd.read_csv('sample-images_01.txt', sep=",", header=None)
data_num = len(df)
df.head()

# 테스트 용 파라미터
K = 2
N = 1


# 베르누이 분포
def bern(x, mu):
    r = 1.0
    for x_i, mu_i in zip(x, mu):
        if x_i == 1:
            r *= mu_i
        else:
            r *= (1.0 - mu_i)
    return r


# 分類結果の表示
def show_figure(mu, cls):
    fig = plt.figure()
    for c in range(K):
        subplot = fig.add_subplot(K, 7, c * 7 + 1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title('Master')
        subplot.imshow(mu[c].reshape(28, 28), cmap=plt.cm.gray_r)
        i = 1
        for j in range(len(cls)):
            if cls[j] == c:
                subplot = fig.add_subplot(K, 7, c * 7 + i + 1)
                subplot.set_xticks([])
                subplot.set_yticks([])
                subplot.imshow(df.iloc[j].values.reshape(28, 28), cmap=plt.cm.gray_r)
                i += 1
                if i > 6:
                    break
    fig.show()


# 초기 매개 변수의 설정
mix = [1.0 / K] * K  # mix = pi_list (size = (K, ))

# TODO 이렇게 수정하면 무슨 문제가 생기는지 알아보기
# mu = (rand(28*28*K)*0.5+0.25).reshape(K, 28*28) # (K,784) Matrix
mu = rand(28 * 28 * K).reshape(K, 28 * 28)  # (size = (K, 284))

# [수정 전]
for k in range(K):
    mu[k] /= mu[k].sum()

# [수정 후]
mu = mu / mu.sum(axis=1)[:, np.newaxis]

fig = plt.figure()

for k in range(K):
    subplot = fig.add_subplot(K, N + 1, k * (N + 1) + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(mu[k].reshape(28, 28), cmap=plt.cm.gray_r)
# fig.show()

# N회의 Iteration을 실시
for iter_num in range(N):

    print("iter_num %d" % iter_num)
    # E phase
    resp = DataFrame()

    # x_n:
    # mu_k :

    # mu_k = mu[k]
    # mu = [mu_0, mu_1, ... mu_k, ... mu_K]
    # X = [x_0, x_1, ... x_n, ... x_N]

    # P(x_n) = sigma_k(pi_k * P_mu_k(x_n))
    # P_mu_k(x_n) = bern(x,mu) = PI_i(mu_k_i ^ x_n_i * ( 1 - mu_k_i)^(1-x_n_i))
    # r_n_k = pi_k * P_mu_k(x_n) / SIGMA_k(pi_k * P_mu_k(x_n))
    # mu_k = SIGMA_n(r_n_k*x_n) / SIGMA_n(r_n_k)
    # pi_k = SIGMA_n(r_n_k)/N

    # mix: pi
    # tmp: r_i_k 길이 K의 행렬
    # resp: r_n_k (데이터 개수, K) 행렬

    # 각각의 이미지 생성기에 소속되는 비율(r_n_k) 계산하기
    for index, line in df.iterrows():  # line: x_n
        tmp = []
        for k in range(K):
            p = bern(line, mu[k])  # p_u_k : p_u_k(x_n)
            a = mix[k] * p  # a: pi * p(x_n)

            if a == 0:
                tmp.append(0.0)
            else:
                s = 0.0
                for kk in range(K):
                    s += mix[kk] * bern(line, mu[kk])  # r의 분모
                tmp.append(a / s)
        resp = resp.append([tmp], ignore_index=True)  # r_n_k_matrix

    # 새로운 이미지 생성기 mu와 이미지 생성기를 선택할 확률 pi 계산하기
    for k in range(K):
        nk = resp[k].sum()  # sigma_n(r_n_k)
        mix[k] = nk / data_num  # pi_k = sigma(n_k) / N
        for index, line in df.iterrows():
            mu[k] += line * resp[k][index]  # mu_k 분모: sigma_n(r_n_k * x_n)
        mu[k] /= nk  # mu_k = sigma_n(r_n_k * x_n) / sigma(r_n_k)

        subplot = fig.add_subplot(K, N + 1, k * (N + 1) + (iter_num + 1) + 1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(mu[k].reshape(28, 28), cmap=plt.cm.gray_r)
    fig.show()

# 트레이닝 세트의 문자를 분류
cls = []
for index, line in resp.iterrows():
    cls.append(np.argmax(line[0:]))

# 분류 결과 표시
show_figure(mu, cls)