# 스칼라(scalar) : 숫자 하나로 이루어진 데이터
# 벡터(vector) : 여러 개의 숫자로 이루어진 데이터 레코드
# 행렬(matrix) : 데이터 레코드가 여러 개 잇는 데이터 집합
import numpy as np
import math

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

A = np.array([[1,2]])
B = np.array([[1,2],[3,4]])
C = np.array([[5],[6]])
#print(np.dot(np.dot(A,B),C))
#print(np.dot(A,np.dot(B,C)))

A = np.arange(1, 10).reshape(3,3)
x = np.array([[1,2,3]])
#print(np.dot(np.dot(x,A),x.T))

A = np.array([[1,1],[1,1]])
x = np.array([-3,5])
#print(np.dot(np.dot(x,A),x.T))

# 행렬 놈(matrix norm)
A = (np.arange(9)-4).reshape((3,3))
#print(np.linalg.norm(A))

# 대각합(trace)
#print(np.trace(np.eye(3)))

# 행렬실(Determinant)
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
det = np.linalg.det(A)

# 역행렬(invers matrix)
a = np.array([[2,0],[0,1]])
#print(np.linalg.inv(a))
b = np.array([[1/math.sqrt(2),-1/math.sqrt(2)],[1/math.sqrt(2),1/math.sqrt(2)]])
#print(np.linalg.inv(b))
c = np.array([[3/math.sqrt(13),-1/math.sqrt(2)],[2/math.sqrt(13),1/math.sqrt(2)]])
#print(np.linalg.inv(c))
d = np.array([[1,1,0],[0,1,1],[1,1,1]])
#print(np.linalg.inv(d))

A = np.array([[1,1,0],[0,1,1],[1,1,1]])
Ainv = np.linalg.inv(A)

# 역행렬과 선형 연립방정식의 해
b = np.array([[2],[2],[3]])
x = np.dot(Ainv,b)

from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
y = boston.target
A = X[:4, [0, 4, 5, 6]]  # 'CRIM', 'NOX', 'RM', 'AGE'
b = y[:4]
#print(np.dot(np.linalg.inv(A),b))

# 최소자승법(least square problem)
A = np.array([[1,1,0],[0,1,1],[1,1,1],[1,1,2]])
b = np.array([[2],[2],[3],[4.1]])
Apinv = np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)
# A의 의사 역행렬
#print(Apinv)
# x값 구하기
x = np.dot(Apinv,b)
#print(x)
# 해를 이용하여 b값 구하기
#print(np.dot(A,x))
#print(b)
# lstsq 명령으로 최소자승문제 구하기
x,resid,rank,s = np.linalg.lstsq(A,b)
#print("x:{}\nresid:{}\nrank:{}\ns:{}\n".format(x,resid,rank,s))
#print(x)
#print(np.dot(A,x))
#print(b)
# 위 코드에서 resid는 잔차벡터의 e = Ax-b의 제곱합, 즉 놈의 제곱이다.

from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
y = boston.target
w,resid,rank,s = np.linalg.lstsq(X,y)
#print(w)

# 벡터의 기하학적 의미
a = np.array([1,2])
#print(np.linalg.norm(a))

# 단위 벡터
a = np.array([1,0])
b = np.array([0,1])
c = np.array([1/np.sqrt(2),1/np.sqrt(2)])
#print(np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c))

# 직교(Orthgonal)
x = np.array([1,1])
y = np.array([1,-1])
z = np.array([-1,1])
#print(np.dot(x.T,y))

# 연습문제 유클리드 거리 및 코사인 거리를 활용한 사용자 유사도 체크 
a = np.array([4,5,2,2])
b = np.array([4,0,3,2])
c = np.array([2,2,0,1])

# 유클리드 거리
def uclid(x,y):
    return math.sqrt(pow(np.linalg.norm(x),2)+pow(np.linalg.norm(y),2)-(2*np.dot(x.T,y)))

# 코사인 거리
def cd(x,y):
    return np.dot(x.T,y)/math.sqrt(np.linalg.norm(x))*math.sqrt(np.linalg.norm(y))

#print(uclid(a,b))
#print(uclid(b,c))
#print(uclid(c,a))

#print(1-cd(a,b))
#print(1-cd(b,c)) 
#print(1-cd(c,a))
    
# 연습문제 프로젝션 성분 벡터와 리젝션 성분 벡터를 구하시오
a = np.array([1,2])
b = np.array([2,0])

def pj(x,y):
    return (np.dot(x.T,y)/pow(np.linalg.norm(y),2))*y

#print("projection attribute : {}".format(pj(a,b)))
#print("rejection : {}".format(a-pj(a,b)))
    
# 행 벡터의 절대값
A = np.arange(4).reshape((2,2))
#print(np.linalg.det(A))

# 선형종속 & 선형독립
x1 = np.array([1,2])
x2 = np.array([3,3])
x3 = np.array([10,14])
#print(2*x1+x2-0.5*x3)

x1 = np.array([1,0])
x2 = np.array([0,1])
# 선형 종속
x1 = np.array([1,0])
x2 = np.array([-1,1])
# 선형 종속
x1 = np.array([1,2])
x2 = np.array([2,4])
#print(2*x1-x2)
# 선형 독립
"""
A = np.array([[1,5,6],[2,6,8],[3,11,14],[1,4,5]])
print(np.linalg.matrix_rank(A))
B = np.array([[1,5,6],[2,6,8],[3,11,14],[1,4,8]])
print(np.linalg.matrix_rank(B))
C = np.array([[1,5,6],[2,6,8],[2,6,8]])
print(np.linalg.matrix_rank(C))
D = np.zeros(9).reshape(3,3)
print(np.linalg.matrix_rank(D))
E = np.array([[3,1,5,7],[3,2,6,10],[9,1,4,6]])
print(np.linalg.matrix_rank(E))
"""

# 역행렬의 존재 유무 판단 -> 행렬식을 확인하여 0일경우 역행렬이 없는것!!!
X = np.array([[1,5,6],[2,6,8],[2,6,8]])
#print(np.linalg.det(X))


# 고윳값(eigenvalue) and 고유벡터(eigenvector)
B = np.array([[2,3],[2,1]])
v1 = np.array([[3],[2]])
#print(np.dot(B,v1))
#print(np.dot(4,v1))
v2 = np.array([[-1],[1]])
#print(np.dot(B,v2))
#print(np.dot(-1,v2))

# 특성방정식(characteristic equation)
A = np.array([[1,-2],[2,-3]])
w1,V1 = np.linalg.eig(A)
#print(w1,V1)
B = np.array([[2,3],[2,1]])
w2,V2 = np.linalg.eig(B)
#print(w2,V2)
C = np.array([[0,-1],[1,0]])
w3,V3 = np.linalg.eig(C)
#print(w3,V3)

V2_inv = np.linalg.inv(V2)
#print(V2.dot(np.diag(w2)).dot(V2_inv))

""" 연습문제 """
X = np.array([[2,3],[2,1]])
e1,v1 = np.linalg.eig(X)
#print(v1.dot(np.diag(e1).dot(np.linalg.inv(v1))))
Y = np.array([[1,1],[0,1]])
e2,v2 = np.linalg.eig(Y)
#print(v2.dot(np.diag(e2)).dot(np.linalg.inv(v2)))

T = np.array([[2,3],[2,1]])
e1,v1 = np.linalg.eig(T)
e = np.array([[4,0],[0,-1]])
v = np.array([[3/math.sqrt(13),2/math.sqrt(13)],[-1/math.sqrt(2),1/math.sqrt(2)]])
#print(np.linalg.inv(v).dot(np.diag(e).dot(v)))
#print(v.dot(np.diag(e).dot(np.linalg.inv(v))))

A = np.array([[60., 30., 20.],
              [30., 20., 15.],
              [20., 15., 12.]])
w,V = np.linalg.eig(A)
w1,w2,w3=w
v1 = V[:,0:1]
v2 = V[:,1:2]
v3 = V[:,2:3]
A1 = v1.dot(v1.T)
A2 = v2.dot(v2.T)
A3 = v3.dot(v3.T)
print(w1*A1)
print(w2*A2)
print(w3*A3)
print(w1*A1+w2*A2+w3*A3)