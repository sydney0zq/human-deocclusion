#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

""" K-Means on the tight boxes, before clustering, please resize them and compress
them into a single npy file."""

import numpy as np
import os
from PIL import Image
import random
import cv2
from sklearn.cluster import MiniBatchKMeans

random.seed(1234)
np.random.seed(1234)

def kmeans_clustering(feats, num_clusters=128, batch_size=5000):
    min_num_samples_per_cluster = 5
    max_num_samples_per_cluster = 2500
    min_num_clusters = 100
    max_num_clusters = 150
    print ("Totally image samples: {}".format(feats.shape[0]))

    while True:
        mbk = MiniBatchKMeans(init='k-means++',
                              n_clusters=num_clusters,
                              batch_size=batch_size,
                              n_init=500,
                              max_no_improvement=50,
                              verbose=1)
        mbk.fit(feats)
        cluster_results = mbk.labels_

        count_list = []
        for k in range(num_clusters):
            count_list.append(np.sum(cluster_results == k))

        count_array = np.array(count_list)
        sort_idx = np.argsort(count_array)
        print ("Count list w.r.t cluster id: {}".format(count_array))
        print ("Count list after sorting: {}".format(count_array[sort_idx]))
        flg1 = np.sum(count_array > min_num_samples_per_cluster) >= min_num_clusters
        flg2 = np.sum(count_array > min_num_samples_per_cluster) <= max_num_clusters
        flg3 = np.max(count_array) < max_num_samples_per_cluster
        print (np.max(count_array))
        
        if flg1 & flg2 & flg3:
            ce = mbk.cluster_centers_
            for i in range(num_clusters):
                cv2.imwrite("cluster_centers/c_{:05d}.png".format(i), ce[i].reshape((32, 32)))
            break
        else:
            print ("*****************************************")
            print (flg1, flg2, flg3)
            print ("Exit condition not satisified, repeat clustering...")

# 这里需要准备将需要聚类的掩码都收集到smDir下面
def read_sm(smDir):
    sms = os.listdir(smDir)
    sm_pool = []
    for sm in sms:
        _sm = np.array(Image.open(os.path.join(smDir, sm)))
        _sm = _sm.reshape(1, -1)
        sm_pool.append(_sm)

    sm_pool = np.concatenate(sm_pool)
    np.save("sm_pool.npy", sm_pool)


print ("Please read code first and then run it.")
exit(0)
#read_sm("shapemask")
sm_pool = np.load("sm_pool.npy")    # Nx1024
kmeans_clustering(sm_pool, 128)




