import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request

# 이미지 불러오기
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/jaehosung/ml4se_codes/main/exercise07/sample-images_01.txt", "sample-images_01.txt"
)

# 트레이닝 셋 할당
df_training_set = pd.read_csv('sample-images_01.txt', sep=",", header=None) # 트레이닝 셋 (size=(600,784))
N = len(df_training_set)
df_training_set.head()

# 베르누이 확률 분포
def bernoulli(x, mu):
    # mu (size = (784,))
    # x (size = (784,))
    p = np.prod(mu[x == 1]) * np.prod(1 - mu[x == 0])
    return p

# 파라미터 설정
K = 3 # 생성기 개수
iteration_num = 1

# 변수 초기화
pi = np.full(K, 1 / K) # pi: K번째 이미지 생성기를 선택할 확률 (size=(K,))
mu = (np.random.rand(28 * 28 * K) * 0.5 + 0.25).reshape(K, 28 * 28) # mu: K개의 이미지 생성기 (size=(K,784))
r_n_k_final = pd.DataFrame()

# TODO mu의 초기화 방식에 대해서 고민해보기 (추측되는 이유: 0에 가까웠을 때?)
# mu = np.random.rand(28 * 28 * classifier_num).reshape(classifier_num, 28 * 28)  # (size = (K, 284))
# mu = mu / mu.sum(axis=1)[:, np.newaxis]

fig_generators, axs_generators = plt.subplots(K, iteration_num+1)  # 생성기의 변화 시각화

# 생성기 시각화
def draw_generator(axs_generators, iter_num, mu):
    for k in range(K):
        axs_generators[k, iter_num].set_xticks([])
        axs_generators[k, iter_num].set_yticks([])
        axs_generators[k, iter_num].imshow(mu[k].reshape(28, 28), cmap=plt.cm.gray_r)


# 이미지 생성기 초기값 시각화
draw_generator(axs_generators, 0, mu)


for iter_num in range(iteration_num):
    print("iter_num %d" % iter_num)

    # expectation (E) step
    # 각각의 이미지 생성기에 소속되는 비율(r_n_k) 계산하기
    r_n_k = pd.DataFrame()

    for _, x_n in df_training_set.iterrows():  # line: x_n (784,)
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
        for i, x_n in df_training_set.iterrows():
            mu_numer[k] += x_n * r_n_k.iloc[i, k]  # mu_k 분모: sigma_n(r_n_k * x_n)
        mu[k] = mu_numer[k] / mu_denom  # mu_k = sigma_n(r_n_k * x_n) / sigma(r_n_k)

        # 생성기 시각화
    draw_generator(axs_generators, iter_num + 1, mu)

    r_n_k_final = r_n_k.copy()

r_argmax = list(r_n_k_final.idxmin(axis = 1)) # 가장 높은 확률로 소속될 생성기의 인덱스


fig_result, axs_result = plt.subplots(K, 7)

# 분류 결과의 표시
def draw_result(mu, r_argmax, axs):
    for c in range(K):
        # 이미지 생성기 시각화
        axs[c,0].set_xticks([])
        axs[c,0].set_yticks([])
        axs[c,0].set_title('Master')
        axs[c,0].imshow(mu[c].reshape(28, 28), cmap=plt.cm.gray_r)

        # K번째 이미지 생성기에 속하는 데이터
        i = 1
        for j in range(len(r_argmax)):
            if r_argmax[j] == c:
                axs[c,i].set_xticks([])
                axs[c,i].set_yticks([])
                axs[c,i].imshow(df_training_set.iloc[j].values.reshape(28, 28), cmap=plt.cm.gray_r)
                i += 1
                if i > 6:
                    break
# 분류 결과 표시
draw_result(mu, r_argmax,axs_result)

fig_result.show()
