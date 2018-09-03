# 스칼라(scalar) : 숫자 하나로 이루어진 데이터
# 벡터(vector) : 여러 개의 숫자로 이루어진 데이터 레코드
# 행렬(matrix) : 데이터 레코드가 여러 개 잇는 데이터 집합
import numpy as np

x1 = np.array([[5.1],[3.5],[1.4],[0.2]])
x2 = np.array([[4.9],[3.0],[1.4],[0.2]])


A = np.array([[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2]])

# 영벡터
zero = np.zeros(10)

# 일벡터
one = np.ones(10)

# 정방행렬(diagonal matrix) : 주 대각행렬에만 숫자가 존재하고 나머지 비 대각 요소가 0인 행렬
temp = np.diag([1,2,3])

# 단위행렬(identity matrix) : 대각행렬중에서 모든 대각성분의 값이 1인 대각행렬을 단위행렬이라고 한다.
I = np.identity(3)

i = np.eye(4)

# 벡터와 행렬의 덧셈과 뺄셈

x = np.array([10,11,12,13,14])
y = np.array([0,1,2,3,4])

# 스칼라를 벡터로 변환하여 연산하는것을 브로드캐스팅(broadcasting)이라고 한다.

# 선형조합(linear combination) : 벡터/행렬에 다음처럼 스칼라 값을 곱한 후 더하거나 뺀 것

# 내적(inner product) : xTy
# 내적은 다음처럼 점(dot)으로 표기하는 경우도 있어서 닷 프로덕트(dot product)라고도 부르고 <  x,y  > 기호로 나타낼 수도 있다.
# x⋅y=<x,y>=xTy
# 두 벡터를 내적하려면 다음과 같은 조건이 만족되어야 한다.
# 1. 우선 두 벡터의 길이가 같아야 한다.
# 2. 앞의 벡터가 행 벡터이고 뒤의 벡터가 열 벡터여야 한다.
# 이 때의 innder product의 값은 scalar값이 되며 다음처럼 계산된다.
x = np.array([[1],[2],[3]])
y = np.array([[4],[5],[6]])
ip = np.dot(x.T,y)
#print(ip)

# 가중합(weighted sum) : 각각의 수에 어떤 가중치 값을 곱하 후 이 곱셈 결과들을 다시 합한것
# 벡터가  x=[x1,⋯,xN]T  이고 가중치 벡터가  w=[w1,⋯,wN]T  이면 데이터 벡터의 가중합은 다음과 같다.
# w1x1+⋯+wNxN=∑i=1Nwixi
stock_price = np.array([[100],[80],[50]])
buy = np.array([[3],[4],[5]])
value = np.dot(stock_price.T,buy)

# 가중평균(weighted average) : 가중합의 가중치값을 전체 가중치값의 합으로 나눈값
value_avg = value/sum(buy)

# 선형회귀모형(linear regression model) : 독립 변수 x에 종속 변수 y를 예측하기 위한 방법의 하나로
# 독립변수 벡터 x와 가중치 벡터 w와의 가중합으로 y에 대한 예측값 y를 계산하는 수식을 말한다.
# y = w1x1 + --- + wnxn
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[1,2],[3,4],[5,6]])
C = np.dot(A,B)

# 연습문제
# 1.순서를 바꾸어  BA 를 손으로 계산하고 NumPy의 계산결과와 맞는지 확인한다.  BA 가  AB 와 같은가? 
D = np.dot(B,A)
# print(C) AB
# print(D) BA --> 결과적으로 같지 않다.
# 2.A , B가 다음과 같을 때, AB, BA를 (계산이 가능하다면) 손으로 계산하고 NumPy의 계산결과와 맞는지 확인한다. AB, BA 모두 계산 가능한가?
A = np.array([1,2,3])
B = np.array([[4,7],[5,8],[6,9]])
#print(np.dot(A,B))
# print(np.dot(B,A))  --> 행과 열이 맞지 않아서 계산이 불가하다.
# 3.A ,  B 가 다음과 같을 때,  AB ,  BA 를 (계산이 가능하다면) 손으로 계산하고
# NumPy의 계산결과와 맞는지 확인한다.  AB ,  BA  모두 계산 가능한가?  BA 의 결과가  AB 와 같은가?
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
C = np.dot(A,B)
D = np.dot(B,A)
#print(C)
#print(D)     --> 결과적으로 같지 않다.
# A 가 다음과 같을 때, AAT와 ATA를 손으로 계산하고 NumPy의 계산결과와 맞는지 확인한다.
# AAT와 AT의 크기는 어떠한가? 항상 정방행렬이 되는가?
A = np.array([[1,2],[3,4],[5,6]])
C = np.dot(A,A.T)
D = np.dot(A.T,A)
#print(C) --> 3행 3열 행렬이 나오고,
#print(D) --> 2행 2열 행렬이 나온다. 따라서 두 행렬의 행과 열이 다르므로 정방행렬이 아니다.
# x 가 다음과 같을 때,  xTx 와  xxT 를 손으로 계산하고 NumPy의 계산결과와 맞는지 확인한다.  xTx 와  xxT 의 크기는 어떠한가?
# 어떤 것이 스칼라이고 어떤 것이 정방행렬인가?
A = np.array([[1],[2],[3]])
C = np.dot(A,A.T)
D = np.dot(A.T,A)
#print(C) --> 3열 3행인 행렬이 나온다.
#print(D)  --> 스칼라 값이 나온다.

# 교환법칙과 분배 법칙
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
C = np.array([[9,8],[7,6]])
# AB != BA
#print(np.dot(A,B))
#print(np.dot(B,A))
# A(B+C) == AB + AC
#print(np.dot(A,(B+C)))
#print(np.dot(A,B)+np.dot(A,C))
# (A+B)T == AT + BT
#print((A+B).T)
#print(A.T+B.T)
# (AB)T == BTAT
#print(np.dot(A,B).T)
#print(np.dot(B.T,A.T))
