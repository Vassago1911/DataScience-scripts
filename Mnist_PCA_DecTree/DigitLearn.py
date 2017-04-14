#!/usr/bin/env python

import os.path
import numpy as np
from pandas import HDFStore,DataFrame,Series,read_csv,read_hdf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def stealPCA(mean, white, components, explainedvar):
    """directly copied all parameters the PCA of sklearn
    uses, now we can just copy the function"""
    def trsf(X):
        X = np.array(X)
        X = X - mean
        X_tfd = np.dot(X,components.T)
        if white:
            X_tfd /= np.sqrt(explainedvar)
        return X_tfd

    return trsf

def savePCAtransform(pc):
    mea = pc.mean_
    whi = pc.whiten
    cps = pc.components_
    exv = pc.explained_variance_

    return (mea, whi, cps, exv)


if (not (os.path.isfile('digipca.h5'))):
    """train.csv hat die Form label; pixel0-783"""
    df = read_csv('train.csv')
    unlabelled = df[df.columns[1:]]
    X = np.array(unlabelled)
    pca = PCA(whiten=True).fit(X)

    l = savePCAtransform(pca)
    l = Series(l)

    X_pca = pca.transform(X)
    df = df.join(DataFrame(X_pca))

    del X
    del X_pca

    dfile = HDFStore('digipca.h5')
    dfile['d1']=df
    dfile['PCA'] = l

    l = list(l)
    pctransform = stealPCA(*l)
else:
    dfile = HDFStore('digipca.h5',)
    df = dfile['d1']
    l = dfile['PCA']
    l = list(l)

    pctransform = stealPCA(*l)

dfile.close(); del dfile
#hiernach ist dfile tot, X, X_pca auch, nur df enthält die Label, Trainingspixel
#und danach die PCA'ten Trainingsdaten

"""jetzt hat schon ein Entscheidungsbaum die faire Chance, auf den most
significant components zu lernen"""

from sklearn import tree
from sklearn.cross_validation import KFold

#for testing
df = df[:300]

kf = KFold(len(df),n_folds = 3)
clfr = list()
scores = list()
for z in kf:
    clfr.append(tree.DecisionTreeClassifier())
    clfr[-1].fit(df.iloc[z[0],:][df.columns[1:]],df.iloc[z[0],:][df.columns[0]])
    scores.append(clfr[-1].score(df.iloc[z[1],:][df.columns[1:]],df.iloc[z[1],:][df.columns[0]]))

print(scores)
#16-08-29:[0.859, 0.855, 0.858, 0.851, 0.862, 0.859, 0.855, 0.857, 0.859, 0.857]
