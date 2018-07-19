"""정수 연산"""
"""
print("{}+{}={}".format(1,1,1+1))
#몫 구하기
print("{}을 {}로 나눈 몫은 {}".format(8,4,8//4))
#나눠진 수 구하기
print("{}을 {}로 나눈 숫자는 {}".format(10,4,10/4))
#나머지 구하기
print("{}을 {}로 나눴을 때 남는 나머지는 {}".format(10,4,10%4))
#제곱 구하기
print("{}의 {}승은 {}".format(2,3,2**3))
#연습 문제
#3×2−8÷4
print((3*2)-(8/4))
#25×6÷3+17
print(((25*6)/3)+17)
#39021−276920÷12040
print(39021-(276920/12040)) 
#26−10%6
print((2**6)-(10%6))"""
"""형변환"""
"""print(int(4.5))
print(float(3))"""
"""자료의 순서만 가지는 리스트 자료형"""
#x=[88,90,100]
#for i in range(len(x)):
#    print(x[i])
#성적 5개에 대한 평균 구하기
#y=[75,45,23,65,98]
#for num in range(5):
#    avg+=y[num]
#print(avg/len(y))
#dic={"math":88,"english":78,"history":45}
#print(dic)
#print("enlgish score is {}".format(dic["english"]))
"""수열 생성하기"""
#0부터 입력값 이전까지의 숫자를 생성한다
#b=list(range(10))
#print(b)
##range(시작값,기준이전값까지 생성,간격)
#c=list(range(1,10))
#print(c)
#d=list(range(1,10,3))
#print(d)
"""자료 추가하기"""
#e=list(range(5))
#print(e)
#e.append(5)
#print(e)
"""자료 삭제하기"""
#f=list(range(5))
#print(f)
#del f[0]
#print(f)
"""슬라이싱"""
#g=list(range(1,21))
#print(g)
#print(g[0:5])
#print(g[-5:-2])
#print(g[-3:])
"""연습문제1"""
"""
#1.리스트에는 숫자 뿐 아니라 문자 등 어떤 값도 넣을 수 있다. 10명으로 이루어진 반의 학생 이름을 생각하여 리스트 변수로 만들어 본다.
student=["강희수","김동완","김민성","김종혁","김한준","류경목","박종범","이승학","위다현","윤여범"]
print(student)
print("student number is {}".format(len(student)))
#2.전학생이 왔다고 가정하여 리스트에 이름을 추가한다.
student.append("이정운")
print(student)
print("student number is {}".format(len(student)))
#3.한 명이 전학을 갔다고 가정하고 리스트에서 이름을 삭제한다.
print(student)
print("{} is gone".format(student[5]))
del student[5]
print(student)
print("student number is {}".format(len(student)))
#4.슬라이싱으로 5번 학생(1번 학생은 가장 처음에 있는 학생이다.)부터 9번 학생까지 5명의 이름을 담은 새로운 리스트를 만든다.
group=student[5:]
print(group)"""
"""리스트와 반복문을 사용하여 계산하기"""
#score=[90, 85, 95, 80, 90, 100, 85, 75, 85, 80]
#print(score)
#print("number:{}".format(len(score)))
#for loop in range(len(score)):
#    avg+=score[loop]
#avg/=len(score)
#print(avg)
#sum=[]
#a1 = [90, 85, 95, 80, 90, 100, 85, 75, 85, 80]
#a2 = [95, 90, 90, 90, 95, 100, 90, 80, 95, 90]
#for i in range(len(a1)):
#    sum.append(a1[i]+a2[i])
#    sum[i]=sum[i]/2
#print(sum)
"""zip함수"""
#두 개의 리스트를 합쳐서 각 리스트 원소의 쌍을 원소로 가지는 하나의 리스트를 말함
#box=list(zip(a1,a2))
#print(box)
#sap=[]
#for a1i,a2i, in zip(a1,a2):
#    sap.append(a1i+a2i)
#print(sap)
"""enumerate함수"""
#enumerate 명령은 리스트의 원소를 반복하면서 동시에 인덱스 값도 생성
#for i, e in enumerate(["a","b","c"]):
#    print("i=%d,e=%s"%(i,e))
#box=list(range(10))
#a1 = [90, 85, 95, 80, 90, 100, 85, 75, 85, 80]
#a2 = [95, 90, 90, 90, 95, 100, 90, 80, 95, 90]
#for i,(a1i,a2i) in enumerate(zip(a1,a2)):
#    box[i]=a1i+a2i
#print(box)
"""리스트의 리스트"""
#X = [[ 85,  90,  20,  50,  60,  25,  30,  75,  40,  55],
#     [ 70, 100,  70,  70,  55,  75,  55,  60,  40,  45],
#     [ 25,  65,  15,  25,  20,   5,  60,  70,  35,  10],
#     [ 80,  45,  80,  40,  75,  35,  80,  55,  70,  90],
#     [ 35,  50,  75,  25,  35,  70,  65,  50,  70,  10]]
#print(X)
#sum=0
#num=0
#for i in range(len(X)):
#    for j in range(len(X[i])):
#        num+=1
#        sum+=X[i][j]
#avg=sum/num
#print("Result of num is {} , sum is {} and average is {}".format(num,sum,avg))
"""연습문제"""
#X=[4,3,2,3,4]
#W=[3,3,1,2,2]
#grade=0
#for i in range(len(X)):
#    grade+=X[i]*W[i]
#    sum=sum+W[i]
#print("grade:{} sum:{}".format(grade,sum))
#print("The average is %.2f"%(float(grade/sum)))
"""연습문제2
자료의 분산(Variance)는 각 자료 값에서 자료의 평균값을 뺀 나머지를 제곱한 값의 평균을 말한다. 예를 들어 자료가 다음과 같다고 하자
X=6,5,4,7,3,5
이 자료의 평균은 다음과 같다
(6+5+4+7+3+5)6=5
각 자료 값에서 자료의 평균값을 뺀 나머지를 제곱한 값을 모두 더한 값의 평균은 다음과 같이 구한다
(6−5)2+(5−5)2+(4−5)2+(7−5)2+(3−5)2+(5−5)26
이 자료의 분산을 구하는 코드를 작성한다."""
X=6,5,4,7,3,5
avg=0
var=0
for loop in range(len(X)):
    avg+=X[loop]
avg/=len(X)
for loop in range(len(X)):
    var+=(X[loop]-avg)**2
var=var/len(X)
print("average is {}\nvariance is {}".format(avg,var))