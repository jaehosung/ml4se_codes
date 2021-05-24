import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# TODO 이미지 다운로드 코드 추가

# 트레이닝 셋 불러오기
df = pd.read_csv('sample-images_01.txt', sep=",", header=None)
N = len(df)
df.head()

# 베르누이 확률 분포
def bernoulli(x, mu):
    # mu (size = (784,))
    # x (size = (784,))
    p = np.prod(mu[x == 1]) * np.prod(1 - mu[x == 0])
    return p


# TODO 이름 다시 정하기
def draw_generator(fig, mu, iter_num, k):
    subplot = fig.add_subplot(K, iteration_num + 1, k * (iteration_num + 1) + (iter_num) + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(mu[k].reshape(28, 28), cmap=plt.cm.gray_r)

# 분류 결과의 표시
def draw_result(mu, cls):
    for c in range(K):
        subplot = fig2.add_subplot(K, 7, c * 7 + 1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title('Master')
        subplot.imshow(mu[c].reshape(28, 28), cmap=plt.cm.gray_r)

        i = 1
        for j in range(len(cls)):
            if cls[j] == c:
                subplot = fig2.add_subplot(K, 7, c * 7 + i + 1)
                subplot.set_xticks([])
                subplot.set_yticks([])
                subplot.imshow(df.iloc[j].values.reshape(28, 28), cmap=plt.cm.gray_r)
                i += 1
                if i > 6:
                    break

# 파라미터 설정
K = 3 # 생성기 개수
iteration_num = 3

# 변수 초기화
pi = np.full(K, 1 / K) # pi: K번째 이미지 생성기를 선택할 확률 (size=(K,))
mu = (np.random.rand(28 * 28 * K) * 0.5 + 0.25).reshape(K, 28 * 28) # mu: K개의 이미지 생성기 (size=(K,784))
r_n_k = pd.DataFrame()

# TODO mu의 초기화 방식에 대해서 고민해보기 (추측되는 이유: 0에 가까웠을 때?)
# mu = np.random.rand(28 * 28 * classifier_num).reshape(classifier_num, 28 * 28)  # (size = (K, 284))
# mu = mu / mu.sum(axis=1)[:, np.newaxis]

fig = plt.figure()  # 생성기의 변화 시각화
fig2 = plt.figure()  # 분류 결과 시각화

# 시작
for k in range(K):
    draw_generator(fig, mu, 0, k)

# 식 정리
# TODO r_n_k 수정하기 중복 선언

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

        # 생성기 시각화
        draw_generator(fig, mu, iter_num + 1, k)

# TODO 이 코드는 이해가 필요함
# 트레이닝 세트의 문자를 분류 결과
cls = []
for _, r_i_k in r_n_k.iterrows():
    cls.append(np.argmax(r_i_k.values))  # np.argmax: ndarray의 최대값의 인덱스

# 분류 결과 표시
draw_result(mu, cls)

fig.show()
fig2.show()