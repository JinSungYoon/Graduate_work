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
#X=6,5,4,7,3,5
#avg=0
#var=0
#for loop in range(len(X)):
#    avg+=X[loop]
#avg/=len(X)
#for loop in range(len(X)):
#    var+=(X[loop]-avg)**2
#var=var/len(X)
#print("average is {}\nvariance is {}".format(avg,var))
"""파이썬 객체지향 프로그래밍"""
#h=10
#v=20
#def area(h,v):
#    return h*v
#a=area(h,v)
#print(a)
"""클래스를 활용하여 객체지향 코딩 예시"""
#class Rectangle(object):
#    def __init__(self,h,v):
#        self.h=h
#        self.v=v
#        
#    def area(self):
#        return self.h*self.v
#r=Rectangle(10,20)
#w=r.area()
#print(w)
"""연습문제1
삼각형의 넓이를 계산하기 위한 클래스를 만든다. 이 클래스는 다음과 같은 속성을 가진다.
밑변의 길이 b 와 높이 h
삼각형의 넓이를 계산하는 메서드 area
"""
#class triangle(object):
#    def __init__(self,h,b):
#        self.h=h
#        self.b=b
#    def area(self):
#        return (self.h*self.b)/2
#    
#t=triangle(14,7)
#test=t.area()
#print(test)
"""연습문제2
사각 기둥의 부피를 계산하기 위한 클래스를 만든다.
이 클래스는 다음과 같은 속성을 가진다.
밑면의 가로 길이 a, 밑면의 세로 길이 b, 높이 h
부피를 계산하는 메서드 volume
겉넓이를 계산하는 메서드 surface
"""
#class figure(object):
#    def __init__(self,a,b,h):
#        self.h=h
#        self.a=a
#        self.b=b
#    def volume(self):
#        return self.a*self.b*self.h
#
#pillar=figure(3,5,8)
#volume=pillar.volume()
#print(volume)    
#class Character(object):
#    def __init__(self):
#        self.life=1000
#        
#    def attacked(self):
#        self.life -=10
#        print(u"{}의 데미지! 생명력:{}".format(10,self.life))
#    def attack(self):
#        print(u"공격!")
#        
#ace=Character()
#bob=Character()
#candy=Character()
#print("세명의 플레이어의 생명력은 \nace:{} bob:{} candy:{}\n입니다.".format(ace.life,bob.life,candy.life))
#ace.attacked()
#bob.attacked()
#candy.attacked()
#ace.attacked()
#bob.attacked()
#bob.attacked()
#ace.attacked()
#print("세명의 플레이어의 생명력은 \nace:{} bob:{} candy:{}\n입니다.".format(ace.life,bob.life,candy.life))
"""클래스 상속"""
#class Warrior(Character):
#    def __init__(self):
#        super(Warrior,self).__init__()
#        self.strength=15
#        self.intelligence=5
#    def attacked(self):
#        self.life-=5
#        print("{}의 데미지! 생명력:{}".format(5,self.life))
#    def attack(self):
#        print(u"육탄 공격!")
#class Wizard(Character):
#    def __init__(self):
#        super(Wizard,self).__init__()
#        self.strength=5
#        self.intelligence=15
#    def attacked(self):
#        self.life-=15
#        print("{}의 데미지! 생명력:{}".format(15,self.life))
#    def attack(self):
#        print(u"마법공격!")
#david=Warrior()
#elice=Wizard()
#print("David의 생명력은 {}\t 힘은 {}\t 지력은 {}이고,\n elice의 생명력은 {}\t 힘은 {}\t 지력은{}입니다.".format(david.life,david.strength,david.intelligence,elice.life,elice.strength,elice.intelligence))
#ace.attack()
#david.attack()
#elice.attack()
#ace.attacked()
#david.attacked()
#elice.attacked()
"""연습문제4
다음과 같이 자동차를 나타내는 Car 클래스를 구현한다.
이 클래스는 최고 속도를 의미하는 max_speed라는 속성과 현재 속도를 나타내는 speed라는 속성을 가진다.
객체 생성시 max_speed 속성은 160이 되고 speed 속성은 0이 된다.
speed_up, speed_down이라는 메서드를 가진다. speed_up을 호출하면 speed 속성이 20씩 증가하고 speed_down을 호출하면 speed 속성이 20씩 감소한다.
스피드 속성 speed의 값은 max_speed 속성 값, 즉 160을 넘을 수 없다. 또 0 미만으로 감소할 수도 없다.
메서드는 호출시 속도 정보를 출력하고 명시적인 반환값을 가지지 않는다.
위 기능이 모두 정상적으로 구현되었음을 보이는 코드를 추가한다.
"""
#class Car(object):
#    def __init__(self):
#        self.max_speed=160
#        self.speed=0
#    def speed_up(self):
#        if(self.speed+20<=self.max_speed):
#            self.speed+=20
#            print("속도 20증가!")
#        else:
#            print("더 이상 속도를 낼 수 없습니다.")
#    def speed_down(self):
#        if(self.speed-20>-1):
#            self.speed-=20
#            print("속도 20감속...")
#        else:
#            print("더이상 감속할 수 없습니다.")
#    def state(self):
#        print("현재 속도는 {}입니다.".format(self.speed))
#audi=Car()
#for i in range(9):
#    audi.speed_up()
#    audi.state()
#for j in range(9):
#    audi.speed_down()
#    audi.state()
"""문자열 인코딩"""
"""파이썬 뿐 아니라 모든 컴퓨터에서 문자는 2진 숫자의 열 즉, 바이트 열(byte sequence)로 바뀌어 저장된다. 이를 인코딩(encoding)이라고 하며 어떤 글자를 어떤 숫자로 바꿀지에 대한 규칙을 인코딩 방식이라고 한다.
가장 기본이 되는 인코딩 방식은 아스키(ASCII) 방식이다.
한글의 경우 과거에는 EUC-KR 방식이 많이 사용되기도 했으나 최근에는 CP949 방식이 더 많이 사용된다."""
#u="가"
#print(len(u))
#b=bytearray("가",'cp949')
#print(len(b))
#u1="ABC"
#u2="가나다"
#print(len(u1),len(u2))
#print(u1[0],u1[1],u1[2])
#print(u2[0],u2[1],u2[2])
#바이트 열이면 글자를 한글자씩 분리할 수 없다.
#b1=bytearray("ABC",'cp949')
#b2=bytearray("가나다",'cp949')
#print(len(b1),len(b2))
#print(chr(b1[0]),chr(b1[1]),chr(b1[2]))
#print(chr(b2[0]),chr(b2[1]),chr(b2[2]))
u="가나다"
print(type(u))
b1=u.encode("cp949")
print(type(b1))
print(b1)
b2=u.encode("euc-kr")
print(type(b2))
print(b2)
b3=u.encode("utf-8")
print(type(b3))
print(b3)
print(b1.decode("cp949"))
print(b2.decode("euc-kr"))
print(b3.decode("utf-8"))