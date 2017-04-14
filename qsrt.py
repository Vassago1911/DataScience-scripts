#!/usr/bin/env python

from random import randint

def testlist(k, ln = 100):
    return [randint(0,k) for i in range(ln)]

def qsrt(l,prstrg = ''):
    if len(l)==0:
        return l
    else:
        li = list(l)
        x = li.pop()
        left = [z for z in li if z<x] 
        mid = [x,]+[z for z in li if z==x]
        right = [z for z in li if z>x]
        print(prstrg,': ', left, ', ', mid, ', ', right)
        return qsrt(left, 'L'+prstrg) + mid + qsrt(right,'R'+prstrg)

for z in range(10):
    qsrt(testlist(100,100))
