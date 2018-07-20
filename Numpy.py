import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import time
"""전체 주석처리는 ctrl+1"""
#"""#numpy로 배열 생성"""
#ar=np.array([0,1,2,3,4,5,6,7,8,9])
#"""print(ar)"""
#"""배열에 새로운 값 대입해서 생성"""
#result=[]
#for size in range(len(ar)):
#    result.append(2*size)
#"""print(result)"""
#"""box=np.array([0,1,2,3,4,5,6,7,8,9])
#box.reshape(2,5)
#print(box)
#"""
#z=np.zeros(10)
#print(z)
#z=np.zeros((2,5))
#print(z)
#"""문자열로 표현"""
#z=np.zeros((3,3),dtype="U4")
#print(z)
#"""1로 배열 채우기"""
#e=np.ones((3,4))
#print(e)
#"""특정값으로 초기화 하지 않은 상태로 배열 생성"""
#r=np.empty((4,3))
#print(r)
#"""규칙적으로 증가하는 배열 생성"""
#inc=np.arange(10)
#print(inc)
#"""처음부터 끝(끝은 미포함) 규칙적인 배열 생성"""
#box=np.arange(3,21,3)
#print(box)
#"""선형 구간 혹은 로그 구간을 지정한 구간의 수만큼 분할"""
#dis=np.linspace(0,100,5)
#print(dis)
#"""행과 열을 바꾸는 연산"""
#A=np.array([[0,1,2],[3,4,5]])
#print(A)
#"""행과 열울 바꿔주는 함수"""
#A=A.T
#print(A)
#"""배열의 크기 변형"""
#box=np.arange(12)
#box=box.reshape(2,6)
#print(box)
#"""reshape에 -1을 넣으면 다른 넣어진 값을 기준으로 재구성"""
#box=box.reshape(3,-1)
#print(box)
#box=box.reshape(2,2,-1)
#print(box)
#"""다시 배열을 1차원으로 만들기 위해서 flatten을 사용"""
#print(box.flatten())
#"""배열에 대해서 차원을 증가시키고 싶을때 newaxis 사용"""
#print(box[:,np.newaxis])
#print(box)
"""#배열의 연결"""
"""행으로 연결할 경우 행의 개수가 일치해야 연결이 가능하다"""
#a1=np.ones((2,3))
#a2=np.zeros((2,2))
#h1=np.hstack([a1,a2])
#print(h1)
"""열로 연결할 경우 열의 개수가 일치해야 연결이 가능하다"""
#b1=np.zeros((2,3))
#b2=np.ones((3,3))
#v1=np.vstack([b1,b2])
#print(v1)
"""dstack에 대한 이해는 필요할듯 하다"""
#c1=np.ones((3,4))
#c2=np.zeros((3,4))
#d1=np.dstack([c1,c2])
#print(d1)
"""각 행렬의 행과 열을 할고 싶다면 shape를 사용"""
#print("v1:{} h1:{} d1:{}".format(v1.shape,h1.shape,d1.shape))
"""2018.07.18"""
"""2차원 그리드 포인트 생성"""
#x=np.arange(3)
#print(x)
#y=np.arange(5)
#print(y)
#X,Y=np.meshgrid(x,y)
#print(X)
#print(Y)
#[list(zip(x,y)) for x,y in zip(X,Y)]
#plt.scatter(X,Y,linewidths=10)
#plt.show()
"""벡터화 연산"""
#x=np.arange(1,10001)
#y=np.arange(10001,20001)
#z=np.zeros_like(x)
#
#for i in range(10000):
#    z[i]=x[i]+y[i]
#print(z)
#a=np.array([1,2,3,4])
#b=np.array([4,2,2,4])
#print(a==b)
#a=np.array([1,2,3,4])
#b=np.array([4,2,2,4])
#c=np.array([1,2,3,4])
#print(np.all(a==b))
#a=np.arange(5)
#print(a)
#print(np.exp(a))
#print(10**a)
#print(np.log(a+1))
"""스탈라 벡터/행렬의 곱셈"""
#x=np.arange(10)
#print(x)
#print(100*x)
#x=np.arange(12).reshape(3,4)
#print(x)
#print(x*100)
"""브로드캐스팅"""
#벡턲리 덧셈 혹은 뺄셈을 하려면 두 벡터의 크기가 같아야 한다.
#그러나 Numpy에서는 서로 다른 크기를 가진 두 배열의 사칙연산 도 지원한다.
#x=np.arange(5)
#y=np.ones_like(x)
#print(x+y)
#print(x+1)
#x=np.vstack([range(7)[i:i+3] for i in range(5)]) 
#print(x)
#y=np.arange(5)[:,np.newaxis]
#print(y)
#print(x+y)
"""차원 축소 연산"""
#x=np.array([1,2,3,4])
#print(x)
#print(np.sum(x))
#print(np.min(x))
#print(np.max(x))
#print(np.argmax(x))     #argmax:최댓값의 위치
#print(np.median(x))
#a=np.zeros((100,100),dtype=np.int)
#print(a)
#x=np.array([[1,1],[2,2]])
#print(x)
#print(np.sum(x))
#print(x.sum(axis=0)) #열(세로) 합계
#print(x.sum(axis=1)) #행(가로) 합계
# 5x6의 데이터 행렬을 만들고 데이터에 대하여
#전체의 최댓값 / 각 행의 합 / 각 열의 평균을 구하시오
#box=np.arange(30).reshape(5,6)
#print(box)
#print(box.max())
#print(box.sum(axis=1))
#print(box.mean(axis=0))
#a=np.array([[4,3,5,7],[1,12,11,9],[2,15,1,14]])
#print(a)
#print(np.sort(a))
#print(np.sort(a,axis=0))
#p=np.array([42,38,12,25])
#j=np.argsort(a) #자료들의 순서만 알고 싶다면 argsort(var)을 사용
#print(j)
#print(p[j])
"""특정 행을 기준으로 정렬하는 방법 찾아야 할듯"""
#rand=np.random.RandomState(50)
#box=rand.randint(0,10,(1,10))
#print(box)
#box=np.partition(box,3)
#print(box)
#student=np.array([[1,2,3,4],[46,99,100,71],[81,59,90,100]])
#print(student)
#print(student[student[:,0].argsort()])
"""기술통계"""
#x=np.array([ 18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
#                2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])
#print("Data count : {}".format(len(x)))   #데이터의 개수
#print("Data mean : {}".format(np.mean(x)))   #샘플의 평균
#print("Sample variance : {}".format(np.var(x)))    #샘플의 분산: 분산이 작으면 데이터가 모여있는것이고, 크면 흩어져 있는것이다.
#print(np.var(x,ddof=1)) #비편향 분산...? 추후 공부한다고 함
#print("Samle standard variance : {}".format(np.std(x)))    #샘플의 표준편차
#print("Sample max value : {}".format(np.max(x)))    #샘플의 최댓값
#print("Sample min value : {}".format(np.min(x)))    #샘플의 최솟값
#print("Sample median value : {}".format(np.median(x)))  #샘플의 중위값
#print("Data min value : {}".format(np.percentile(x,0)))   #최솟값
#print("Data One quartile : {}".format(np.percentile(x,25)))     #1사분위수
#print("Data two quartile : {}".format(np.percentile(x,50)))     #2사분위수
#print("Data three quartile : {}".format(np.percentile(x,75)))   #3사분위수
#print("Data four quartile : {}".format(np.percentile(x,100)))   #최댓값
##SciPy 패키지에는 여러가지 기술 통계 값을 한번에 구해주는 describe 명령이 있다.
#from scipy.stats import describe
#print(describe(x))
"""난수 발생과 카운팅"""
#print(np.random.seed(0))
#print(np.random.rand(5))
#print(np.random.rand(10))
"""데이터 순서 바꾸기"""
#x=np.arange(10)
#print(x)
#np.random.shuffle(x)
#print(x)
"""데이터 샘플링"""
"""
numpy.random.choice(a, size=None, replace=True, p=None)
a : 배열이면 원래의 데이터, 정수이면 range(a) 명령으로 데이터 생성
size : 정수. 샘플 숫자
replace : 불리언. True이면 한번 선택한 데이터를 다시 선택 가능
p : 배열. 각 데이터가 선택될 수 있는 확률
"""
#print(np.random.choice(5,3,replace=False)) #5개중 3개만 선택
#print(np.random.choice(5,10))   #반복해서 10개 선택
#print(np.random.choice(5,10,p=[0.1,0,0.3,0.5,0.1]))    #선택 확률을 다르게 해서 10개 선택
#print(np.random.rand(10))
#print(np.random.rand(3,5))      #(세로,가로)
"""numpy.random.randint(low, high=None, size=None)
만약 high를 입력하지 않으면 0과 low사이의 숫자를, high를 입력하면 low와 high는 사이의 숫자를 출력한다. size는 난수의 숫자이다."""
#print(np.random.randint(10,size=10))    #10미만의 10개의 데이터 출력
#print(np.random.randint(10,20,size=10)) #10이상 20미만의 데이터 10개 생성
#print(np.random.randint(10,20,size=(3,5)))
"""정수 데이터 카운팅"""
#print(np.unique([11,11,2,2,34,34]))
#a=np.array(['a','b','b','c','a'])
#index,count=np.unique(a,return_counts=True) #중복된을 제외한 값을 index에 저장하고 각 요소의 개수를 count에 저장한다.
#print("index : {}".format(index))
#print("count : {}".format(count))
"""특정 범위안의 수의 개수를 카운팅 할 경우 bincount minlength"""
#print(np.bincount([1,1,2,2,3],minlength=6))