import numpy as np
#
#
rand_list = np.asarray(range(4))
# np.random.seed(42)
# def train_datagen(epoch_num=5):
#     while (True):
#         # n_count = 0
#         # if n_count == 0:
#         np.random.shuffle(rand_list)
#         print(rand_list)
#             # print(n_count)
#             # n_count = 1
#         for _ in range(epoch_num):
#             for i in range(4):
#                 num = rand_list[i]
#                 yield num
#
# td = train_datagen(epoch_num=5)
# for i in range(40):
#     n_p = next(td)
#     print(n_p)
import time


def ret1(num):
    return num+2, num*3


def ret2():
    x = 1+2
    y = 2*3
    return x, y


def train_datagen(epoch_num=5):
    while (True):
        # n_count = 0
        # if n_count == 0:
        np.random.shuffle(rand_list)
        print(rand_list)
            # print(n_count)
            # n_count = 1
        for _ in range(epoch_num):
            for i in range(4):
                num = rand_list[i]
                yield ret1(num)


td = train_datagen(epoch_num=5)
for i in range(40):
    x_p, y_p = next(td)
    print(x_p, y_p)