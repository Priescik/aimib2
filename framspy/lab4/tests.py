import random
import numpy as np

# A = np.ones((4,10,2))
# B = np.zeros((4,10))
# A[3,9,1] = -1
# B[2,8] = -1
# Amask = (A == -1) | (B == -1)[:,:,None]

# # print(res) #.sum(axis=(2)))
# print(Amask)





# a = [[None for _ in range(2)] for a in range(3)]
# a[2][0] = 1
# print(a)




# a = [random.random() for _ in range(10)]
# a = sorted(a)
# print(a)
# h = np.histogram(a, bins=5)
# print(h[0])
# print(h[1])


parent_fits = [0, 1, 1, 4, 4, 5, 7, 10]


hist, brackets = None, None
for bins in range(5, 5*2, 1):
    hist, brackets = np.histogram(parent_fits, bins=bins)
    if (hist > 0).sum() >= 5:
        break


index = 0
while len(parent_fits)-1 > index: #
    hist_idx = np.digitize(parent_fits[index], brackets, right=True) - 1
    if hist_idx == -1:
        hist_idx = 0
    if hist[hist_idx] > 1:
        del parent_fits[index]
        hist[hist_idx] -= 1
    else:
        index += 1
