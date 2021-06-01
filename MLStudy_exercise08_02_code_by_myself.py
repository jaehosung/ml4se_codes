import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 실제 분포
beta_true = 1.0 / (0.3) ** 2  # 분산: 0.3
alpha = 1.0 / (100) ** 2  # 사전 분포 분산

M = 9  # 다항식 차수
N_list = [4, 5, 10, 100]

# 트레이닝 셋 {x_n,y_n} (n=1...N) 생성 함수
def create_training_set(num):
    dataset = pd.DataFrame(columns=['x','y'])
    for i in range(num):
        x = float(i) / float(num-1)  # x의 범위: [0,1]
        y = np.sin(2.0 * np.pi * x) + np.random.normal(scale=0.3)
        dataset = dataset.append(pd.Series([x,y], index = ['x','y']),
                                 ignore_index=True)
    return dataset

# dataset =
phis = pd.DataFrame() # (size=(N*M+1))

num = 4
m = 4

training_set = create_training_set(num)
phis = pd.DataFrame()
for i in range(0,m+1):
    p = training_set.y ** i
    p.name = "x^%d" % i
    phis = pd.concat([phis,p],axis=1)

print(phis)
