import pandas as pd
import numpy as np

# raise ValueError('')

def generate_dataset(N):
    dataset = pd.DataFrame(columns=["x","t"])
    for i in range(N):
        x = float(i) / float(N-1)
        y = np.sin(2.0 * np.pi * x) + np.random.normal(scale=0.3)
        dataset = dataset.append(pd.Series([x,y], index=["x","t"]),
                                 ignore_index=True)
    return dataset

def func_mx(args):
    x = training_set.x
    t = training_set.t
    phi_x = np.array([x ** i for i in range(0,M+1)])
    sigma_t_phi= np.sum(np.multiply(t.values[:,np.newaxis],phis.values),axis = 0)
    m_x = np.linalg.multi_dot([beta*phi_x.T, s, sigma_t_phi]).flatten()
    return m_x


def func_sx(args):
    pass

def draw_t_distribution(c, training_set, func_mx, func_sx):
    pass


beta_true = 1.0 / (0.3) ** 2
beta = beta_true
alpha = 1.0 / 100 ** 2

M = 9  # 다항식 차수
N_list = [4, 5, 10, 100]

for c, N in enumerate(N_list):
    training_set = generate_dataset(N)
    draw_t_distribution(c,training_set, func_mx, func_sx)