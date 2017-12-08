import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from math import log
from bitarray import bitarray


def targ_per_city(df, city , col='city'):
    df = df.dropna()
    sum_of_city = df[df[col] == city]['msno'].count()
    targ_1 = df[(df[col] == city) & (df['target'] == 1)]['msno'].count()
    return (targ_1 / sum_of_city)


def targ_per_age(df, age, col='bd'):
    df = df.dropna()
    sum_of_age = df[df[col] == age]['msno'].count()
    targ_1 = df[(df[col] == age) & (df['target'] == 1)]['msno'].count()
    return (targ_1 / sum_of_age)

def bitarray2IntList(df):
    dfcat = []
    incr = len(df.index)/100
    i = 0
    for index, row in df.iterrows():
        i += 1
        print (i)
        catln = bitarray()
        for j in range(len(row)):
            catln += eval(row[j])
        catln = [int(j) for j in catln]
        dfcat.append(catln)
        if index % incr == 0 :print (index // incr, '%')
    dfcat = np.array(dfcat)
    return dfcat

def featureIntEncode(df,Dropnan = True):
    if Dropnan : df.dropna()
    lst = []
    for i in df.columns:
        tmp2 = []
        tmp2 = pd.DataFrame(df[i].unique())
        lst.append(tmp2)
    return lst

def featureBinaryEncode(df, saveFlag=False, savePath='', fileName=''):
    df = df.dropna()
    l = df.size
    max_bits = ceil(log(l, 2))
    col_enc = []
    for index, row in df.iterrows():
        tmp = bitarray("{0:0{1}b}".format((index+1), (max_bits)))
        col_enc.append(tmp)
    col_name = fileName+'-bin'
    df[col_name] = pd.Series(col_enc, index=df.index)
    if (saveFlag):
        df.to_csv(savePath+fileName+'.csv', sep=',', index=False)
    return df

def addOneHot(df):
    return df.join(pd.get_dummies(df))
