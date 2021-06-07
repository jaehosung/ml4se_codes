'''
TODO
    1. 주석 달기
    2. flatten 이유 찾기
    3. colab에 옮기기
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_dataset(N):
    dataset = pd.DataFrame(columns=["x", "t"])
    for i in range(N):
        x = float(i) / float(N - 1)
        y = np.sin(2.0 * np.pi * x) + np.random.normal(scale=0.3)
        dataset = dataset.append(pd.Series([x, y], index=["x", "t"]),
                                 ignore_index=True)
    return dataset


def calc_s(phis):
    phiphi_sum = 0
    for _, phi in phis.iterrows():
        phi = phi.values[:, np.newaxis]  # reshape the phi (M+1,) to (M+1,1)
        phiphi_sum += np.dot(phi, phi.T)  # (size=(M+1,M+1))
    s_inv = alpha * pd.DataFrame(np.identity(M + 1)) + beta * phiphi_sum
    s = np.linalg.inv(s_inv)
    return s


def calc_mean(t, s, phis):
    sigma_t_phi = np.sum(np.multiply(t.values[:, np.newaxis], phis.values),
                         axis=0)
    mean = np.linalg.multi_dot([beta * s, sigma_t_phi])
    return mean


def calc_phis(training_set):
    phis = pd.DataFrame()  # (size=(N,M+1)
    for i in range(0, M + 1):
        phi = training_set.x ** i
        phi.name = "x**%d" % i
        phis = pd.concat([phis, phi], axis=1)
    return phis


def calc_parameters(training_set, M):
    t = training_set.t

    phis = calc_phis(training_set)  # 모든 training_set의 phi값 모음
    s = calc_s(phis)  # s 계산
    mean = calc_mean(t, s, phis)  # mean 계산

    def func_mx(x):
        phi_x = np.array([x ** i for i in range(0, M + 1)])
        sigma_t_phi = np.sum(np.multiply(t.values[:, np.newaxis], phis.values),
                             axis=0)
        # TODO 왜 flatten? maybe 차수 때문에?
        mx = np.linalg.multi_dot([beta * phi_x.T, s, sigma_t_phi]).flatten()
        return mx

    def func_sx(x):
        phi_x = np.array([x ** i for i in range(0, M + 1)])
        variance = 1.0 / beta + np.linalg.multi_dot([phi_x.T, s, phi_x])
        return np.sqrt(variance.diagonal())

    return mean, s, func_mx, func_sx


def draw_t_distribution(ax, training_set, func_mx, func_sx):
    line_x = np.arange(0, 1.01, 0.01)
    line_y_true = np.sin(2 * np.pi * line_x)

    # 평균과 표준편차 곡선 계산
    line_m = np.array(func_mx(line_x))
    line_s = np.array(func_sx(line_x))

    # 그래프 이름 및 범위 설정
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-2, 2)

    ax.set_title("N=%d" % N)
    ax.scatter(training_set.x, training_set.t, marker='o', color='blue')
    ax.plot(line_x, line_y_true, color='green', linestyle=':')  # 실제 값

    # 평균과 표준편차 곡선 표시
    ax.plot(line_x, line_m, color='red', label='mean')
    ax.plot(line_x, line_m - line_s, color='black', linestyle='--')
    ax.plot(line_x, line_m + line_s, color='black', linestyle='--')
    ax.legend(loc=1)


def draw_ws(ws_samples, func_mx, M):
    line_x = np.arange(0, 1.01, 0.01)
    line_y_m = np.array(func_mx(line_x))

    axs[c][1].set_xlim(-0.05, 1.05)
    axs[c][1].set_ylim(-2, 2)
    axs[c][1].set_title("N=%d" % N)
    axs[c][1].scatter(training_set.x, training_set.t, marker='o', color='blue')
    # 평균 곡선 표시
    axs[c][1].plot(line_x, line_y_m, color='red', label="mean")
    axs[c][1].legend(loc=1)

    def f(x, ws):  # x: (size=(n,1)), ws: (size=(M+1))
        x_n = np.array([x ** i for i in range(0, M + 1)])
        return np.multiply(ws.values[:, np.newaxis], x_n).sum(axis=0)

    for index, ws in ws_samples.iterrows():
        line_y_f = f(line_x, ws)
        axs[c][1].plot(line_x, line_y_f, color='green', linestyle='--')


beta_true = 1.0 / (0.3) ** 2
beta = beta_true
alpha = 1.0 / 100 ** 2

M = 9  # 다항식 차수
N_list = [4, 5, 10, 100]

fig, axs = plt.subplots(len(N_list), 2, figsize=(8, len(N_list) * 4))
for c, N in enumerate(N_list):
    training_set = generate_dataset(N)
    mean, s, func_mx, func_sx = calc_parameters(training_set, M)
    draw_t_distribution(axs[c][0], training_set, func_mx, func_sx)

    ws_sample = pd.DataFrame(np.random.multivariate_normal(mean, s, 4))
    draw_ws(ws_sample, func_mx, M)
fig.show()
