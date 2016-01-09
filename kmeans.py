#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===================================================================
#     FileName: kmeans.py
#       Author: bruc14
#   CreateTime: 2016-01-08 00:13
# ===================================================================
from sklearn.metrics import euclidean_distances
from sklearn.datasets import load_iris
from sklearn.cluster import k_means
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
from scipy.io.arff import loadarff
from sklearn.datasets.samples_generator import make_blobs
from time import time
import pdb

def kmeans (X,k,centers):
    dist_s = euclidean_distances(X,centers)
    label = dist_s.argmin(axis=1)
    while len(set(label)) < k:
        centers=np.random.randn(k,X.shape[1])
        centers = MinMaxScaler().fit_transform(centers)
        dist_s = euclidean_distances(X,centers)
        label = dist_s.argmin(axis=1)
    n_iter=0
    while True:
        print("n_iter:",n_iter)
        centers_new = [X[label==i].mean(axis=0) for i in range(k)]
        # pdb.set_trace()
        if np.equal(centers_new, centers).all():
            break
        else:
            centers = centers_new
            dist_s = euclidean_distances(X,centers)
            label = dist_s.argmin(axis=1)
        n_iter += 1
    return label, n_iter


if __name__ == '__main__':
    # data=load_iris()
    # X, y=data.data,data.target
    k=5
    X, y = make_blobs(n_samples=int(1e4), centers=k, n_features=5, random_state=0)
    # X=[
            # [1,2],
            # [1,3],
            # [2,4],
            # [-5,-6],
            # [-1,-5],
            # [-6,-9]
            # ]
    # y=[0,0,0,1,1,1]
    X = MinMaxScaler().fit_transform(X)
    # centers = [random.random()*X.max(axis=0) for i in np.arange(k)]
    centers=np.random.randn(k,X.shape[1])
    centers = MinMaxScaler().fit_transform(centers)
    t0 = time()
    plabel,n_iter = kmeans(X, k, centers)
    ari = adjusted_rand_score(y, plabel)
    print("%-30s%-30s%-30s%-30s"%("My",ari,time()-t0,n_iter))

    to=time()
    centroid,label,inertia, n_iter  = k_means(X, k, init = centers, return_n_iter=True)
    ari = adjusted_rand_score(y, label)
    print("%-30s%-30s%-30s%-30s inertia:%s"%("sklearn",ari,time()-t0,n_iter, inertia))
