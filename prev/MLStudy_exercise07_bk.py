import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request

# # 이미지 불러오기
# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/jaehosung/ml4se_codes/main/exercise07/sample-images_01.txt",
#     "sample-images_01.txt")
#
# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/jaehosung/ml4se_codes/main/exercise07/sample-labels_01.txt",
#     "sample-labels_01.txt")

# 트레이닝 세트 읽기

# 트레이닝 셋 불러오기
df = pd.read_csv('sample-images_01.txt', sep=",", header=None)
N = len(df)
df.head()
# ==========
# parameters
# ==========
K = 3
iteration_num = 3

# 베르누이 분포
# mu (size = (784,))
# x (size = (784,))

# [수정 전]
# def bernoulli(x, mu):
#     r = 1.0
#     for x_i, mu_i in zip(x, mu):
#         if x_i == 1:
#             r *= mu_i
#         else:
#             r *= (1.0 - mu_i)
#     return r

# [수정 후]
def bernoulli(x, mu):
    # mu (size = (784,))
    # x (size = (784,))
    p = np.prod(mu[x == 1]) * np.prod(1 - mu[x == 0])
    return p


# TODO 이름 다시 그리기
def draw_plot(fig, mu, iter_num, k):
    subplot = fig.add_subplot(K, iteration_num + 1, k * (iteration_num + 1) + (iter_num + 1) + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(mu[k].reshape(28, 28), cmap=plt.cm.gray_r)

# pi: K 이미지 생성기를 선택할 확률 (size=(K,))
# mu: 총 K개의 이미지 생성기 (size=(K,784))

# 초기 매개 변수의 설정
# [수정 전]
# pi = [1.0 / classifier_num] * classifier_num  # mix = pi_list (size = (K, ))

# [수정 후]
pi = np.full(K, 1 / K)

# TODO 이렇게 수정하면 무슨 문제가 생기는지 알아보기
mu = (np.random.rand(28 * 28 * K) * 0.5 + 0.25).reshape(K, 28 * 28)  # (K,784) Matrix
# mu = np.random.rand(28 * 28 * classifier_num).reshape(classifier_num, 28 * 28)  # (size = (K, 284))

# [수정 전]
# for k in range(classifier_num):
#     mu[k] /= mu[k].sum()
# [수정 후]
# TODO 이 코드가 왜 필요한지 고민해보기 안넣으면 에러 뜸
# mu = mu / mu.sum(axis=1)[:, np.newaxis]

fig = plt.figure()
for k in range(K):
    subplot = fig.add_subplot(K, iteration_num + 1, k * (iteration_num + 1) + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(mu[k].reshape(28, 28), cmap=plt.cm.gray_r)

# 식 정리
'''
# x_n:
# mu_k :

# mu_k = mu[k]
# mu = [mu_0, mu_1, ... mu_k, ... mu_K]
# X = [x_0, x_1, ... x_n, ... x_N]

# P(x_n) = sigma_k(pi_k * P_mu_k(x_n))
# P_mu_k(x_n) = bernoulli(x,mu) = PI_i(mu_k_i ^ x_n_i * ( 1 - mu_k_i)^(1-x_n_i))
# r_n_k = pi_k * P_mu_k(x_n) / SIGMA_k(pi_k * P_mu_k(x_n))
# mu_k = SIGMA_n(r_n_k*x_n) / SIGMA_n(r_n_k)
# pi_k = SIGMA_n(r_n_k)/N

# mix: pi
# tmp: r_i_k 길이 K의 행렬
# resp: r_n_k (데이터 개수, K) 행렬
# r_i_k: k번째 분류기로부터 이미지 x_i가 얻어질 가능성
'''

# TODO r_n_k 수정하기 중복 선언
r_n_k = pd.DataFrame()

for iter_num in range(iteration_num):
    print("iter_num %d" % iter_num)

    # expectation (E) step
    # 각각의 이미지 생성기에 소속되는 비율(r_n_k) 계산하기
    r_n_k = pd.DataFrame()

    for _, x_n in df.iterrows():  # line: x_n (784,)
        r_i_k = []
        for k in range(K):
            p = bernoulli(x_n, mu[k])  # p_u_k : p_u_k(x_n)
            r_i_k_numer = pi[k] * p  # a: pi * p(x_n) / r_i_k의 분자

            if r_i_k_numer == 0:
                r_i_k.append(0.0)
            else:
                r_i_k_denom = 0.0  # sigma(pi * P_mu_k)
                for kk in range(K):
                    r_i_k_denom += pi[kk] * bernoulli(x_n, mu[kk])  # r_i_k의 분모
                r_i_k.append(r_i_k_numer / r_i_k_denom)
        r_n_k = r_n_k.append([r_i_k], ignore_index=True)  # r_n_k (size=(N,K))

    # maximization (M) step
    # 새로운 이미지 생성기 mu와 이미지 생성기를 선택할 확률 pi 계산하기
    mu_numer = np.zeros(28 * 28 * K).reshape(K, 28 * 28)  # (size = (K, 284))

    for k in range(K):
        mu_denom = r_n_k[k].sum()  # sigma_n(r_n_k) dtype: float
        pi[k] = mu_denom / N  # pi_k = sigma(n_k) / N
        for i, x_n in df.iterrows():
            mu_numer[k] += x_n * r_n_k.iloc[i, k]  # mu_k 분모: sigma_n(r_n_k * x_n)
        mu[k] = mu_numer[k] / mu_denom  # mu_k = sigma_n(r_n_k * x_n) / sigma(r_n_k)

        draw_plot(fig, mu, iter_num, k)
    fig.show()

# 트레이닝 세트의 문자를 분류
cls = []
for _, r_i_k in r_n_k.iterrows():
    cls.append(np.argmax(r_i_k[0:]))

# 분류 결과의 표시
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


# 분류 결과 표시
show_figure(mu, cls)
