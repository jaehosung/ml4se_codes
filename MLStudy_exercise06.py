import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from PIL import Image # PIL: Python Imaging Library
import urllib.request 

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/jaehosung/ml4se/main/photo.jpg", "flower.jpg"
)
im = Image.open("flower.jpg")
print("size:", im.size)  # 520x346 픽셀의 이미지
plt.imshow(im)

# 이미지 데이터 [r,g,b]의 w*h길이의 1차원 배열을 image_data에 저장
image_data = list(im.convert("RGB").getdata()) 
    # 179920(520x346) 개의 (r,g,b) 튜플 list
print("first five pixels:", image_data[:5])

def fit_k_means(image_data, K, MAX_ITERATION_NUM):
    def set_initial_mu():
        mu_initial_list = np.random.randint(256,size=(K,3)) # [[r,g,b],...[r,g,b]]
        #TODO
        # print("Initial centers:", mu_initial_list)
        print("Initial centers:", [x.tolist() for x in mu_initial_list])
        print("========================")
        
        return mu_initial_list
    def calc_new_mu_list(mu_list):
        def calc_closest_cluster():
            min_sq_dist = 256*256*3 #각 데이터의 제곱 에러값을 최대로 설정
            k = -1
            for i in range(K):
                mu_i = mu_list[i]
                sq_dist = sum([x_i * x_i for x_i in x - mu_i])
                if sq_dist < min_sq_dist:
                    min_sq_dist = sq_dist
                    k = i
            return k, min_sq_dist
        
        for pixel_idx, x in enumerate(image_data):
            k, min_sq_dist = calc_closest_cluster(x, mu_list)
            mu_idx_list[pixel_idx] = k
            x_sum_list[k] += x
            x_num_list[k] += 1
            J_new += min_sq_dist
        
        for i in range(K):
            if x_num_list[i] != 0:
                mu_list_new[i] = np.array(x_sum_list[i] / x_num_list[i], dtype = int)
        return mu_list_new, J_new, mu_idx_list, x_num_list
            

    def print_log(iter_num, mu_list, x_num_list, J_new):
        print("itration: %d" % (iter_num + 1))
        print("New centers:", end=" ")
        print([x.tolist() for x in mu_list])
        print("Num of data in each cluster:", end=" ")
        print(x_num_list)
        print("J=%d" % J_new)
        print("========================")

    def check_stop_point(J_new,J_old):
        if iter_num > 0 and J - J_new < J * 0.001:
            print("Iteration ends")
            return true
        else:
            return false

    mu_list = set_initial_mu()
    
    J_old = 0
    for iter_num in range(MAX_ITERATION_NUM):
        mu_list, J_new, mu_idx_list, x_num_list = calc_new_mu_list(mu_list)
        if check_stop_point(J_new,J_old) == True:
            break
        J_old = J_new      
    clustered_image_data = calc_clustered_image_data()
    err = calc_err(mu_list)

    return mu_list, mu_idx_list

# 파라미터 설정
cluster_num = 3
iteration_num = 10
print("========================")
print("Number of clusters: K=%d" % cluster_num)
print("========================")

mu_list, mu_idx_list = fit_k_means(image_data, cluster_num, iteration_num)
print(mu_list)
def fit_k_means(image_data, K, MAX_ITERATION_NUM):
    def set_initial_mu():
        mu_initial_list = np.random.randint(256,size=(K,3)) # [[r,g,b],...[r,g,b]]
        #TODO
        # print("Initial centers:", mu_initial_list)
        print("Initial centers:", [x.tolist() for x in mu_initial_list])
        print("========================")
        
        return mu_initial_list
    def calc_new_mu_list(mu_list):
        def calc_closest_cluster():
            min_sq_dist = 256*256*3 #각 데이터의 제곱 에러값을 최대로 설정
            k = -1
            for i in range(K):
                mu_i = mu_list[i]
                sq_dist = sum([x_i * x_i for x_i in x - mu_i])
                if sq_dist < min_sq_dist:
                    min_sq_dist = sq_dist
                    k = i
            return k, min_sq_dist
        
        for pixel_idx, x in enumerate(image_data):
            k, min_sq_dist = calc_closest_cluster(x, mu_list)
            mu_idx_list[pixel_idx] = k
            x_sum_list[k] += x
            x_num_list[k] += 1
            J_new += min_sq_dist
        
        for i in range(K):
            if x_num_list[i] != 0:
                mu_list_new[i] = np.array(x_sum_list[i] / x_num_list[i], dtype = int)
        return mu_list_new, J_new, mu_idx_list, x_num_list
            

    def print_log(iter_num, mu_list, x_num_list, J_new):
        print("itration: %d" % (iter_num + 1))
        print("New centers:", end=" ")
        print([x.tolist() for x in mu_list])
        print("Num of data in each cluster:", end=" ")
        print(x_num_list)
        print("J=%d" % J_new)
        print("========================")

    def check_stop_point(J_new,J_old):
        if iter_num > 0 and J - J_new < J * 0.001:
            print("Iteration ends")
            return true
        else:
            return false

    mu_list = set_initial_mu()
    
    J_old = 0
    for iter_num in range(MAX_ITERATION_NUM):
        mu_list, J_new, mu_idx_list, x_num_list = calc_new_mu_list(mu_list)
        if check_stop_point(J_new,J_old) == True:
            break
        J_old = J_new      
    clustered_image_data = calc_clustered_image_data()
    err = calc_err(mu_list)

    return mu_list, mu_idx_list

# 파라미터 설정
cluster_num = 3
iteration_num = 10
print("========================")
print("Number of clusters: K=%d" % cluster_num)
print("========================")

mu_list, mu_idx_list = fit_k_means(image_data, cluster_num, iteration_num)
print(mu_list)

# mu_initial_list = generate_random_rgb(cluster_num) 
# mu_list, mu_idx_list = calc_k_means(image_data, mu_initial_list, cluster_num, iteration_num) 
# image_data_clustered = calc_clustered_image_data(image_data, mu_list, mu_idx_list)
