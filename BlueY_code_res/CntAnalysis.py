import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def mae_err(l):
    return l.sum()/len(l)

def plotsquare(squ):
    fig, ax = plt.subplots()
    ax.imshow(squ, cmap = plt.cm.gray, interpolation = 'nearest')
    plt.show()

dfday = pd.read_csv('day.csv'); dfhour = pd.read_csv('hour.csv')

#in the interest of not producing the primitive and useless model
#casual + registered = cnt, I just delete these
del dfday['casual']; del dfday['registered'];
del dfhour['casual']; del dfhour['registered'];

#by my first exploration with gretl I know that the only weekday
#which passed (0.95-)significance was Saturday
dfday.loc[:,'Saturday']=(dfday['weekday']==6).map(int)
dfhour.loc[:,'Saturday']=(dfhour['weekday']==6).map(int)

#I want these tables to scream that they'll be fitted with a regression
dfday['const'] = 1; dfhour['const'] = 1;

del dfday['weekday']; del dfhour['weekday']

#furthermore I shall resort to the primitive conversion of dates
#to first day - last day by integer numbers, so they have a numeric
#representation, which is, however, very specific to this fixed
#dataset
swap = lambda t: (t[1],t[0])
#the enumerator produces list of tuples (index, value), I want
#to map value to index, so I swap the tuple
l = dict(map(swap,list(enumerate(dfday.dteday.values))))
#now the map l can convert the dates to integers saying n-th day in the set
dfday = dfday.replace({'dteday': l}); dfhour = dfhour.replace({'dteday': l})
#let's check with a type-test that we indeed mapped all days and there aren't any
#date strings left
assert all(map(lambda i: isinstance(i, np.int64), dfday.dteday.values))
assert all(map(lambda i: isinstance(i, np.int64), dfhour.dteday.values))

#here is the random 10% split you requested
dfday, dfdayholdout = train_test_split(dfday, test_size = 0.1)
dfhour, dfhourholdout = train_test_split(dfhour, test_size = 0.1)

#now we can get an indication of relevance by looking at the
#correlation-table

#plotsquare(abs(dfday.corr())); plotsquare(abs(dfhour.corr()))

#we find big areas of pitch-black, so the linear relationships to
#cnt are actually quite pronounced, just as the pre-exploration in
#Gretl indicated

predindicesday = ['dteday', 'season',
 'yr', 'mnth', 'holiday',
'workingday', 'weathersit', 'temp', 'atemp','hum', 'windspeed', 'Saturday']
predindiceshour = ['dteday', 'season', 
'yr', 'mnth', 'hr', 'holiday',
'workingday', 'weathersit', 'temp', 'atemp','hum', 'windspeed', 'Saturday']

#I love having a visual indication of how my data is distributed, so
#I usually resort to a PCA, no matter what I am doing
pcaday = PCA(whiten=True,n_components=2).fit(dfday[predindicesday].values)
pcahour = PCA(whiten=True,n_components=2).fit(dfhour[predindiceshour].values)

princday = pcaday.transform(dfday[predindicesday].values)
princhour = pcahour.transform(dfhour[predindiceshour].values)

#Anzeigen der Tages-Daten in einer Grafik
#plt.figure(figsize=(8, 6))
#plt.scatter(princday[:,0], princday[:,1])
#plt.show()

#<-- I'm confused -- there is obviously something highly significant and relevant
#going on here that we have those two clusters, which just look like a broken and
#reordered line-segment. I thought it was due to the "wrong" numbering of seasons,
#but that did not change anything.

#Anzeigen der Stunden-Daten in einer Grafik
#plt.figure(figsize=(8, 6))
#plt.scatter(princhour[:,0], princhour[:,1])
#plt.show()

#By looking at pcaday.explained_variance_ratio_ and pcahour.explained_variance_ratio_,
#we find 99.98% and 0.02% (rounded to two decimals obviously) and 99.8% and 0.01%,
#looking at pcaday.components_ and pcahour.components_ we find that the property
#explaining 99.98% of the variance is the first one - i.e. the date. So I could improve
#the analysis much, by removing the general trend-observation that the number of
#users grew in time, probably owing to the fact that this bike rental was new and
#still grew just by new users noticing they exist.

#Finally as promised let's do the linear regression - I got really stuck trying to
#get scikit-learn to work with additional p-values, so by frustration I went to
#'statsmodels', which I did not know before this :)
import statsmodels.api as sm

#I had a big multicollinearity-problem with the daymodel, so I preprocess that
#design matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor

crit = 10
l = [15]
while max(l)>crit:
    l = list()
    for i in range(len(dfday[predindicesday].columns)):
        l.append(variance_inflation_factor(dfday[predindicesday].values,i))
    if max(l)>crit:
        del predindicesday[np.argmax(l)]

oneinsignificant = True
#we're doing the backwards elimination once more

predindicesday = ['const']+predindicesday

while (predindicesday and oneinsignificant): #i.e., while nonempty          
    daymodel = sm.OLS(dfday.cnt.values, dfday[predindicesday])
    resultsday = daymodel.fit()
    #the [1:] is to make sure we do not accidentally throw the constant
    #out, where the significance check is not relevant
    ttest = resultsday.pvalues.drop('const')
    oneinsignificant = (np.max(ttest) >= 0.05)
    if oneinsignificant:
        delindex = np.argmax(ttest);
        predindicesday = [z for z in predindicesday if z!=delindex]

crit = 10
l = [15]
while max(l)>crit:
    l = list()
    for i in range(len(dfhour[predindiceshour].columns)):
        l.append(variance_inflation_factor(dfhour[predindiceshour].values,i))
    if max(l)>crit:
        del predindiceshour[np.argmax(l)]

oneinsignificant = True
#we're doing the backwards elimination once more

predindiceshour = ['const']+predindiceshour

while (predindicesday and oneinsignificant): #i.e., while nonempty          
    hourmodel = sm.OLS(dfhour.cnt.values, dfhour[predindiceshour])
    resultshour = hourmodel.fit()
    #the [1:] is to make sure we do not accidentally throw the constant
    #out, where the significance check is not relevant
    ttest = resultshour.pvalues.drop('const')
    oneinsignificant = (np.max(ttest) >= 0.05)
    if oneinsignificant:
        delindex = np.argmax(ttest);
        predindiceshour = [z for z in predindiceshour if z!=delindex]

#============= The MAE-Evaluation!! =============
def maerror(mdl, rsltmdl, df, target):
    l = np.array(df[target])
    r = np.array(rsltmdl.predict(df[mdl.exog_names]))
    assert len(l) == len(r)
    r1 = abs(l-r)
    return sum(r1)/len(r1)

print(maerror(daymodel, resultsday, dfdayholdout, 'cnt'))
print(maerror(hourmodel, resultshour, dfhourholdout, 'cnt'))
