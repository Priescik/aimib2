# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import time
import numpy

# a = numpy.asarray([[10.4391, 9.7712,  0.3597],
#  [ 8.9975,  9.503,  -0.0091],
#  [ 9.9985,  9.506,  -0.023 ],
#  [10.9994,  9.4992, -0.0587],
#  [11.0238,  9.4636,  0.8377],
#  [11.0487,  9.4275,  1.8005],
#  [11.0025, 10.4979, -0.0211],
#  [10.0035, 10.501,  -0.0082]]).T
# # Sample data
# x = a[0]
# y = a[1]
# z = a[2]

# # Create a 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z)

# # Labeling axes
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_xlim(8, 12)  # Set limits for the X-axis
# ax.set_ylim(8, 12)  # Set limits for the Y-axis
# ax.set_zlim(-1, 3)  # Set limits for the Z-axis

# # Show the plot
# plt.show()

#######################

# a = numpy.asarray([[[10.4391, 9.7712,  0.3597],
#  [ 8.9975,  9.503,  -0.0091],
#  [ 9.9985,  9.506,  -0.023 ],
#  [10.9994,  9.4992, -0.0587],
#  [11.0238,  9.4636,  0.8377],
#  [11.0487,  9.4275,  1.8005],
#  [11.0025, 10.4979, -0.0211],
#  [10.0035, 10.501,  -0.0082]]])
# # a = numpy.random.rand(1,10000,3)
# b = numpy.random.rand(1,100,3)
# c = numpy.random.rand(100,1,3)
# d = b-c
# print(d.shape)


# p1 = a
# p2 = a.reshape((a.shape[1],1,3))	
# P = p1 - p2	
# dist = numpy.linalg.norm(P, axis=2)
# print(dist)

# print(time.perf_counter())
# dist1 = 0
# dist2 = 0
# for i in a[0, 1:]:
#     dist = numpy.linalg.norm(a[0, 1:]-i, axis=1)
#     # print(dist)
#     dist1 = max(dist.max(), dist1)

# print(time.perf_counter())
# dist1 = 0
# dist2 = 0
# for i in a[0, 1:]:
#     for j in a[0, 2:]:
#         dist = numpy.linalg.norm(j-i)
#         dist1 = max(dist, dist1)
# print(time.perf_counter())
# for i in a[-1, 1:]:
#     for j in a[-1, 2:]:
#         dist = numpy.linalg.norm(j-i)
#         dist2 = max(dist, dist2)
# fitness = [dist2/dist1]


# for i in a[-1, 1:]:
#     for j in a[-1, 2:]:
#         dist = numpy.linalg.norm(j-i)
#         dist2 = max(dist, dist2)
# fitness = [dist2/dist1]

s = time.perf_counter()
a=0
for i in range(30000000):
    a+=1
print(time.perf_counter()-s)