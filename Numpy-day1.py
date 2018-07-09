import numpy as np
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
a1=np.ones((2,3))
a2=np.zeros((2,2))
h1=np.hstack([a1,a2])
print(h1)
"""열로 연결할 경우 열의 개수가 일치해야 연결이 가능하다"""
b1=np.zeros((2,3))
b2=np.ones((3,3))
v1=np.vstack([b1,b2])
print(v1)
"""dstack에 대한 이해는 필요할듯 하다"""
c1=np.ones((3,4))
c2=np.zeros((3,4))
d1=np.dstack([c1,c2])
print(d1)
"""각 행렬의 행과 열을 할고 싶다면 shape를 사용"""
#print("v1:{0} h1:{1} d1:{2}".format(v1,h1,d1)
