#!/usr/bin/env python

import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, SparsePCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from collections import Counter

def forma(st):
    #print(st)
    sttemp = ''.join(filter(lambda x: x in ' abcdefghijklmnopqrstuvwxyzäöüß', st))
    return sttemp

df = pd.read_csv('nokiasms15-0816.csv',sep=";",
                 names=['b0','b1','number','b3','b4','date','b6','message'])
df = df[['number','date','message']]

"""ich weiß über diese Daten, dass die String-Länge im Mittel 138 ist mit
Sigma^2=228, also würfeln wir die erstmal raus"""
df['stringstat'] = df['message'].apply(len)
df = df[df['stringstat']<=900]
"""Die Handynummern, die überhaupt vorkommen, sind exponentiell (exp(-t))
verteilt, nur 5 haben mehr als 100, nur 15 mehr als 10 SMS auf sich.. also
werde ich auch die rauswürfeln!"""
d = dict(Counter(df['number']))
d = {k: v>=10 for k,v in d.items()}

df['numbercpy'] = df['number']
df['numbercpy'].replace(d,inplace=True)
df = df[df['numbercpy']]
df = df[['number','date','message']]
"""Reindex s.t. the indices are contiguous again"""
df = pd.DataFrame(list(df.values))
df.columns = ['number','date','message']

#l=list(df.index); r=list(range(len(l))); random.shuffle(r);
#l=dict(zip(l,r)); df.rename(l, inplace=True);
#df.sort_index(inplace=True)

df['message'] = df['message'].apply(str.lower)
df['message'] = df['message'].apply(forma)
df['message'] = df['message'].apply(str.strip)
#print(df.head())

dfvised = CountVectorizer().fit_transform(df['message'].values)
X = dfvised.copy()
#pca = PCA(n_components = 2, whiten=True).fit(X.todense())
pca = PCA(whiten=True).fit(X.todense())
#spca = SparsePCA(n_components = 3).fit(X.todense())
X_pca = pca.transform(X.todense())

d = pd.DataFrame(X_pca[:,0:4])
d = d.join(df['number'])

#X_spca = spca.transform(X.todense())
#fig = plt.figure(figsize = (8,6))
#ax = fig.add_subplot(111, projection='3d')
#ax1.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2])
#ax.scatter(X_spca[:,0],X_spca[:,1],X_spca[:,2])
#plt.show()


#ine = list()
#Y=X.todense()
#for K in range(6,8):
#	model = KMeans(n_clusters=K)

#	mdl = model.fit(scale(Y))
#	ine.append(mdl.inertia_)

	# Anzeigen der Daten in einem Grafik
fig = plt.figure(figsize=(16, 12))
g = sns.pairplot(d,hue="number")
plt.show()

#cp1 = fig.add_subplot(121)
#cp2 = fig.add_subplot(122)
#cp3 = fig.add_subplot(223)
#cp4 = fig.add_subplot(224)

#cp1.scatter(X_pca[:,0], X_pca[:,1])
#cp1.ylabel("Principal Components")
#cp2.scatter(X_pca[:,-2],X_pca[:,-1])
#cp2.xlabel('Least Components')
#cp3.scatter(X_pca[0,:],X_pca[1,:])
#cp4.scatter(X_pca[-2,:],X_pca[-1,:])
#plt.show()

#for i in range(len(ine)-1):
#	ine[i] = ine[i]-ine[i+1]
"""now we have list of improvements in error"""

#plt.scatter(range(len(ine)),ine)
#plt.show()
