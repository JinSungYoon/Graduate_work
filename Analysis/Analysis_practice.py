import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import math
from collections import Counter

# 랜덤seed값 설정 
np.random.seed(0)

# 랜덤하게 친구의 수 생성
friends = [100,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

# 친구의 숫자를 숫자별로 counting
friend_counts = Counter(friends)

# x축에 100개의 숫자 생성 / y축에 count된 친구들 배열로 생성
xs = range(51)
ys = [friend_counts[x] for x in xs]

# 바형태로 그래프 그리기
plt.bar(xs,ys)
plt.axis([0,101,0,25])
plt.title("histogram of friend  Counts")
plt.xlabel("# of friends")
plt.ylabel("# of people")
plt.show()

# 총 수집한 데이터의 개수 파악
num_points = len(friends)

# 가장 큰 값과 가장 작은값
largest_value = max(friends)
smallest_value = min(friends)
print("number : {} / max : {} / min : {}".format(num_points,largest_value,smallest_value))

# 데이터를 작은 순서대로 배열하여 가장 작은 값과 두번째로 작은 값 
sorted_values = sorted(friends)
smallest_value = sorted_values[0]
second_smallest = sorted_values[1]
middle_value = sorted_values[int(len(sorted_values)/2)]
second_largest = sorted_values[-2]

#print(sorted_values)
print("smallest : {} / second_smallest : {} / middle_values : {} / second_largest : {} / largest : {}".format(smallest_value,second_smallest,middle_value,second_largest,largest_value))

# 데이터의 평균값 
def mean(x):
    return sum(x)/len(x)
print("friends_mean : {}".format(mean(friends)))

# 데이터의 중위값
def median(x):
    n = len(x)
    sorted_v = sorted(x)
    midpoint = n // 2
    # 데이터가 만일 짝수라면
    if len(x)%2==1:
        return sorted_v[midpoint] 
    else:
        lo = midpoint-1
        hi = midpoint
        return (sorted_v[lo]+sorted_v[hi])/2
print("friends_median : {}".format(median(friends)))

# 4분위값
def quantile(x,p):
    p_index = int(p*len(x))
    return sorted(x)[p_index]

print("low 10% : {}".format(quantile(friends,0.1)))
print("low 25% : {}".format(quantile(friends,0.25)))
print("low 75% : {}".format(quantile(friends,0.75)))
print("low 90% : {}".format(quantile(friends,0.9)))

# 가장 많이 나온 갯수 파악
print("Most common frined number best5 : {}".format(Counter(friends).most_common(5)))

# 가장 큰 값과 작은 값의 차이
def data_range(x):
    return max(x)-min(x)

print("Gap of largest number and smallest number : {}".format(data_range(friends)))

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

print("Variance of data : {}".format(variance(friends)))

def standard_deviation(x):
    return math.sqrt(variance(x))

print("Standard_deviation of data : {}".format(standard_deviation(friends)))

# 상관관계 

def covariance(x,y):
    n = len(x)
    return np.dot(de_mean(x),de_mean(y))/(n-1)

daily_minutes = daily_minutes = daily_minutes = [1,68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]

# 공분산이 양수이면 x의 값이 클수록 y의 값이 크고, x의 값이 작을수록 y의 값이 작아진다는 의미이다.
# 공분산이 음수이면 x의 값이 클수록 y의 값이 작고, x의 값이 작을수록 y의 값이 커진다는 의미이다.
# 공분산이 0이면 그와 같은 관계가 존재하지 않는다는 이야기.
print("Covariance : {}".format(covariance(friends,daily_minutes)))

def correlation(x,y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x,y) / stdev_x / stdev_y
    else:
        return 0
    
# 상관관계는 단위가 없으며, -1(완벽한 음의 상관관계)에서 1(완벽한 양의 상관관계) 사이의 값을 갖는다.
# 예를 들어 상관관계가 0.25라면 상대적으로 약한 양의 상관관계를 의미한다.
print("Correlation : {}".format(correlation(friends,daily_minutes)))

plt.plot(friends, daily_minutes, 'r+',alpha=0.5)
plt.axis([0,max(friends)+10,0,max(daily_minutes) +10 ])
plt.show()
# 100명이라는 친구가 이상치이기 때문에 index에 100을 넣은것이다.
outlier = friends.index(100)

friends_good = [x
                for i, x in enumerate(friends)
                if i != outlier]
daily_minutes_good = [x
                      for i, x in enumerate(daily_minutes)
                      if i != outlier]

# 이상치를 제거했을때의 상관관계
print("Correlation : {}".format(correlation(friends_good,daily_minutes_good)))
plt.plot(friends_good,daily_minutes_good,'g+',alpha=0.5)
plt.axis=[0,max(friends_good)+10,0,max(daily_minutes_good)+10]
plt.show()