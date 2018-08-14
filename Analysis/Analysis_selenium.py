import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

# 한글 폰트 안 깨지게하기위한 import
import matplotlib.font_manager as fm

# 가져올 폰트 지정
font_location='E:/글꼴/H2GTRE.TTF'
# 폰트 이름 지정 
font_name=fm.FontProperties(fname=font_location).get_name()
mpl.rc('font',family=font_name)

kovo_result_table = pd.read_pickle('Kovo_result_table')

# 데이터의 평균값 
def mean(x):
    return sum(x)/len(x)

def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

# 모든 값들에 제곱을 해서 더한다.
def sum_of_squares(x):
    return sum([x_i**2 for x_i in x])

def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n-1)

def standard_deviation(x):
    return math.sqrt(variance(x))

def covariance(x,y):
    n = len(x)
    return np.dot(de_mean(x),de_mean(y))/(n-1)

def correlation(x,y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x,y) / stdev_x / stdev_y
    else:
        return 0

#print(kovo_result_table[('공격','성공률')])
print("Covariance : {}".format(covariance(kovo_result_table[('공격','범실')],kovo_result_table[('공격','순위')])))

plt.plot(kovo_result_table[('공격','범실')],kovo_result_table[('공격','순위')],'r+',alpha=0.5)
plt.axis([0,max(kovo_result_table[('공격','범실')])+10,0,max(kovo_result_table[('공격','순위')])+10])
plt.show()
