import matplotlib.pyplot as plt
import numpy as np
from CifarLoader import CifarLoader


class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i) for i in range (1,6)]).load()
        self.test = CifarLoader(["test_batch"]).load()


# def display_cifar(images, size):
#     n = len(images)
#     plt.figure()
#     plt.gca().set_axis_off()
#     im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
#                     for i in range(size)])
#     # plt.imshow(im)
#     # plt.show()
#
#
#
#
# d = CifarDataManager()
# print ("Number of train images: {}".format(len(d.train.images)))
# print ("Number of train labels: {}".format(len(d.train.labels)))
# print ("Number of test images: {}".format(len(d.test.images)))
# print ("Number of test images: {}".format(len(d.test.labels)))
# images = d.train.images
# display_cifar(images, 10)