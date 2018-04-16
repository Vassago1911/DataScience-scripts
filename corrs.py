import pandas as pd, os

def reduce(l, fun):
    res = l.pop()
    for x in l:
        res = fun(res,x)
    return res

def list_sub(l,x):
    return [y for y in list(l) if y!=x]

files = list(filter(lambda x: x.find('.csv')>-1,os.listdir()))
ccy_pairs = list(map(lambda x: x.split('_')[2],files))

dfs = [pd.read_csv(f,header=None, sep=';') for f in files]

for i in range(len(dfs)):
    dfs[i].columns = ['time',ccy_pairs[i],'high','low','close','vol']
    dfs[i]['time'] = pd.to_datetime(dfs[i].time, format='%Y%m%d %H%M%S')
    dfs[i] = dfs[i][['time',ccy_pairs[i]]]

mrg = lambda x,y: x.merge(y,how='outer',on='time').fillna(method='ffill').fillna(method='bfill').fillna(0)
df = reduce(dfs,mrg)
val_cols = list_sub(df.columns,'time')

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

models = list()

for c in val_cols[:1]:
    independent_cols = list_sub(val_cols, c)
    dependent_col = [c,]
        
    for d in independent_cols:
        ind_cols = list_sub(independent_cols,d)
        ind = df[ind_cols]
        dep = df[dependent_col]

        #dreckig!!
        dep = dep.shift(-600).dropna()
        ind = ind[:-600]
        
        X = add_constant(ind)
        y = dep
        
        vifs = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)

        models.append((sm.OLS(y,X).fit(),vifs))    

for m in models:
    print('=================================')
    print(m[0].summary2())
    print(m[1])
    print('=================================')

"""
d = df.set_index('time')
res = pd.DataFrame()
shift = 600
for c in d.columns:
    print(c)
    ddf = d.copy(deep=True)
    ddf['shift_'+str(shift)] = ddf[c].shift(-shift)
    tmp = ddf.dropna().corr()['shift_600']
    print(tmp)
    res[c+'_shift_'+str(shift)] = round(tmp,2)
"""

