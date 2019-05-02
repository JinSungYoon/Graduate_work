#import plotly.plotly as py
#import plotly.graph_objs as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
import collections

MData = pd.read_pickle("E:/대학교/졸업/졸업작품/분석연습/Extract_M_Data")
FData = pd.read_pickle("E:/대학교/졸업/졸업작품/분석연습/Extract_F_Data")

# 분산이 더 큰 그룹 찾기 (결과는 남자 그룹의 분산이 더 작다)
Mp_team = Data[Data['플레이오프진출']==1]
Mf_team = Data[Data['플레이오프진출']==0]
Fp_team = Data[Data['플레이오프진출']==1]
Ff_team = Data[Data['플레이오프진출']==0]

P_male = np.arange(len(Mp_team.index))
F_male = np.arange(len(Mf_team.index))
P_female = np.arange(len(Fp_team.index))
F_female = np.arange(len(Ff_team.index))

for loop in range(len(Data.columns)):
    var_name = Data.columns[loop]
    # 만일 여자선수의 분산이 남자 선수의 분산보다 클경우
    if(np.var(FData[var_name])>np.var(MData[var_name])):
        print("{0} {1}".format("female",np.var(FData[var_name])-np.var(MData[var_name])))
    else:
        print("{0} {1}".format("male",np.var(MData[var_name])-np.var(FData[var_name])))

z = np.abs(stats.zscore(MData))

MQ1 = Mdata.quantile(0.10)
MQ3 = Mdata.quantile(0.90)
FQ1 = Fdata.quantile(0.10)
FQ3 = Fdata.quantile(0.90)

IQR = Q3-Q1
#print(IQR)

# 이상치가 더 많은 그룹 찾기(결과는 남자가 이상치가 더 많다)
mcount=0
fcount=0
#for i in range(len(MData.columns)):
#    var_name = Mdata.columns[i]
#    for j in range(len(MData[var_name])):
#        if((MData[var_name][j]>=MQ1[var_name]) and (MData[var_name][j]<=MQ3[var_name])):
#            mcount+=1
#    for k in range(len(FData[var_name])):
#        if((FData[var_name][k]>=FQ1[var_name]) and (FData[var_name][k]<=FQ3[var_name])):
#            fcount+=1
#    # 남자 데이터의 이상치가 더 많으면
#    if(len(MData[var_name])-mcount>len(FData[var_name])-fcount):
#        print("{0} {1}".format("male",(len(MData[var_name])-mcount)-(len(FData[var_name])-fcount)))
#    else:
#        print("{0} {1}".format("female",(len(FData[var_name])-fcount)-(len(MData[var_name])-mcount)))
#    fcount=0
#    mcount=0

# 남자와 여자데이터 분산 그래프 표현
        
#for loop in range(len(Data.columns)):
#    var = Data.columns[loop]
#    plt.title(var)
#    plt.scatter(P_male,Mp_team[var],marker='s',c='b',linestyle='solid')
#    plt.scatter(F_male,Mf_team[var],marker='*',c='r',linestyle='solid')
##    plt.scatter(P_female,Fp_team[var],c='g',linestyle='solid')
##    plt.scatter(F_female,Ff_team[var],c='y',linestyle='solid')
#    plt.xticks(rotation=90)
#    plt.figure(figsize=(12,10))
#    plt.show()
#    
#for loop in range(len(Data.columns)):
#    var = Data.columns[loop]
#    plt.title(var)
##    plt.scatter(P_male,Mp_team[var],marker='s',c='b',linestyle='solid')
##    plt.scatter(F_male,Mf_team[var],marker='*',c='r',linestyle='solid')
#    plt.scatter(P_female,Fp_team[var],c='g',linestyle='solid')
#    plt.scatter(F_female,Ff_team[var],c='y',linestyle='solid')
#    plt.xticks(rotation=90)
#    plt.figure(figsize=(12,10))
#    plt.show()