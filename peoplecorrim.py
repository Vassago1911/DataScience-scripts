import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def swap(l,i,j):
    l[i],l[j]=l[j],l[i]

def plotsquare(squ):
    fig, ax = plt.subplots()
    ax.imshow(squ, cmap = plt.cm.gray, interpolation = 'nearest')
    plt.show()

def rex(squ,l):
    return squ.reindex_axis(l,axis=0).reindex_axis(l,axis=1)

dfile = pd.HDFStore('RedHat2.h5')

#people = pd.read_csv('./zipdata/people.csv')
people = dfile.people
re = [z for z in people.columns if (type(people[z].iloc[0])==np.bool_)]
#now re contains the columns with boolean entries, the others are all strings,
#except char_38 which is a genuine integer larger than 1

pp = people[re].copy()

square = pp.corr()

sq = square.transpose()
#sq = sq.sort_values('char_10',axis=0)
#l = ['char_29', 'char_14', 'char_31', 'char_12', 'char_30', 'char_27',
#     'char_33', 'char_35', 'char_26', 'char_18', 'char_25', 'char_24',
#     'char_11', 'char_20', 'char_15', 'char_34', 'char_22', 'char_37',
#     'char_32', 'char_21', 'char_28', 'char_19', 'char_13', 'char_17',
#     'char_36', 'char_23', 'char_16', 'char_10']

#sq=rex(sq,l)
plotsquare(sq)

"""a simple correlation-analysis leads me to believe that char_14, char_18, 
char_21, char_25 and char_31 should be sufficient as independent characteristics, .. 
the correlation matrix was basically one big block sum A\oplus B, and there even is
a variable correlated with another by 0.97 --> we don't need all this .."""

if ('char_10' in people.columns):
    people = people[['people_id', 'char_1', 'group_1', 'char_2', 'date', 'char_3',
                    'char_4', 'char_5', 'char_6', 'char_7', 'char_8', 'char_9',
                    'char_38','char_14','char_18','char_21','char_25','char_31']]
    dfile['people'] = people

dfile.close()

#TODO: da ist ne Exponentialverteilung in char_38 drin (people['char_38'].hist hat
# an Null nen Ausreißer, Rest wächst exponentiell ( vielleicht nur quadratisch, jedenfalls
# mit Krümmung ) .. try: Sortieren nach char_38, und dann finden, welcher
# linear mitwächst)
#TODO.1: ok, nur quadratisch -- s. char38ohne0, der Counter findet linearen Zuwachs, wenn
#eins die Null rauswirft, und hist summiert dann ja noch auf (bins und so), d.h. wir kriegen
#sowas wie Summe über erste n natürliche Zahlen ~ O(n^2)
#    wie krieg ich jetzt den Parameter, mit dem eins char_38 auch abhängig kriegt?
