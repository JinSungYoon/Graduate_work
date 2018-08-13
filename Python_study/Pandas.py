"""<2018.07.24>"""
import pandas as pd
import numpy as np
s= pd.Series([9904312,3448737,2890451,2466052],index=["Seoul","Busan","Incheon","Daegue"])
#print(s)
#print(s.index)
#print(s.values)
#s.name="인구"
#s.index.name="도시"
#print(s.index.name)
#시리즈에 연산을 하면 value에만 적용된다
#print(s/100000)
#print(s[(250e4<s)&(s<500e4)])
#Pandas에서는 뒤에 나오는 숫자까지 포함하므로 주의해야한다.
#print(s[:3])
#s0=pd.Series(range(3),index=["a","b","c"])
#print(s0)
#print("서울" in s)
#for k,v in s.items():
#    print("%s=%d"%(k,v))
s2=pd.Series({"Seoul":9631482,"Busan":3393191,"Incheon":2632035,"Daejoen":1490158})
#print(s2)
#딕셔너리의 원소는 순서를 가지지 않으므로 시리지의 데이터도 순서가 보장되지 않는다.
#만약 순서를 정하고 싶다면 인덱스를 리스트로 지정해야한다.
s2=pd.Series({"Seoul":9631482,"Busan":3393191,"Incheon":2632035,"Daejeon":1490158},
             index=["Busan","Seoul","Incheon","Daejeon"])
#print(s2)
"""인덱스 기반 연산"""
ds=s-s2
#print(ds)
#print(s.values-s2.values)
#print(ds.notnull())
#print(ds[ds.notnull()])
#rs=(s-s2)/s2*100
#rs=rs[rs.notnull()]
#print(rs)
"""데이터 수정"""
#rs["Busan"]=1.63
#print(rs)
##데이터 추가
#rs["Daegue"]=1.41
#print(rs)
##데이터 삭제
#del rs["Seoul"]
#print(rs)
#volleyball=pd.Series({"receive":76.1,"spike":42.7,"toss":65.3,"dig":22.7,"attack":52.3,"defense":42.75},
#                    index=["attack","spike","defense","dig","receive","toss"])
#volleyball.name="KEPCO"
#print(volleyball)
#soccer=pd.Series({"pass":65.2,"counterattack":24.5,"defense":67.2,"attack":45.2,"shot":42.2,"tackle":12.4},
#                 index=["attack","counterattack","shot","pass","defense","tackle"])
#soccer.name="Mancity"
#print(soccer)
#log=volleyball-soccer
#print(log)
"""데이터프레임 클래스"""
data={
    "2015": [9904312, 3448737, 2890451, 2466052],
    "2010": [9631482, 3393191, 2632035, 2431774],
    "2005": [9762546, 3512547, 2517680, 2456016],
    "2000": [9853972, 3655437, 2466338, 2473990],
    "지역":["수도권","경상권","수도권","경상권"],
    "2010-2015 증가율":[0.0283,0.0163,0.0982,0.0141]
    }
columns=["지역","2015","2010","2005","2000","2010-2015 증가율"]
index=["서울","부산","인천","대구"]
df=pd.DataFrame(data,index=index,columns=columns)
#print(df)    
#열방향 인덱스와 행방향 인덱스 붙히기
df.index.name="도시"
df.columns.name="특성"
#print(df)
result={
        "Point":[100,81,77,75,70],
        "Win":[32,25,23,21,21],
        "Draw":[4,6,8,12,7],
        "Lose":[2,7,7,5,10],
        "Goal difference":[79,40,38,46,24]}
items=["Point","Win","Draw","Lose","Goal difference"]
Team_name=["MCI","MUN","TOT","LIV","CHE"]
league=pd.DataFrame(result,index=Team_name,columns=items)
#print(league)
#데이터 프레임에 T를 붙혀서 전치(Transpose)를 하는것이 가능하다.
#print(league.T)
#print(league[["Win","Draw","Lose"]])
df2=pd.DataFrame(np.arange(12).reshape(3,4))
#print(df2)
df["2010-2015 증가율"]=df["2010-2015 증가율"]*100
#print(df)
#print(df[1:3])
data={
      "Korea":[80,90,70,30],
      "English":[90,70,60,40],
      "Math":[90,60,80,70],}
columns=["Korea","English","Math"]
index=["Kim","Lee","Park","Choi"]
df=pd.DataFrame(data,columns=columns,index=index)
#print(df)
#1.모든 학생의 수학 점수를 시리즈로 나타낸다.
#print(df[["Math"]])
#2.모든 학생의 국어와 영어 점수를 데이터 프레임으로 나타낸다.
#print(df[["English","Korea"]])
#3.모든 학생의 각 과목 평균 점수를 새로운 열로 추가한다.
#axis=1이 행 기준으로 평균을 구하라는 의미로 해석
avg=df.mean(axis=1)
df["Average"]=avg
#print(df)
#4.Choi의 영어 점수를 80점으로 수정하고 평균 점수도 다시 계산한다.
#df.loc["Choi","English"]=80
#print(df)
#avg=df.mean(axis=1)
#df["Average"]=avg
#print(df)
#문제 해결해야 한다.
#Kim의 점수를 데이터프레임으로 나타낸다.
#print(df.iloc[0])
#Park의 점수를 시리즈로 나타낸다.
#print(df.iloc[2])
"""데이터프레임 인덱서"""
box=pd.DataFrame(np.arange(10,22).reshape(3,4),
                index=["r1","r2","r3"],
                columns=["c1","c2","c3","c4"])
#print(box)
"""loc인덱서"""
#df.loc[행인덱스(row),열인덱스(column)]와 같은 형태로 사용한다.
#print(box.loc["r1","c2"])
#print(box.loc["r1":,"c3"])
#print(box.loc["r2":,"c2":])
#특정 조건에 해당하는 것만 추출
#print(box.loc[box.c1>10])
#print(box.loc["r1",:])
#print(box[:1])
#열 데이터 추가
#box["c5"]=[14,18,22]
#print(box)
#행 데이터 출가
#box.loc["r4"]=[90,91,92,93,94]
#print(box)
#행 데이터 추가 / 제거
#box.loc["r5"]=[100,101,102,103,104]
#print(box)
#box=box.drop("r5")
#print(box)
box2=pd.DataFrame(np.arange(10,26).reshape(4,4),
                  columns=np.arange(1,8,2))
#print(box2)
#print(box2.loc[1,1])
#print(box2.loc[1:2,:])
"""iloc인덱서"""
#정수 인덱스만 방는다
#box의 0행 1열 데이터
#print(box.iloc[0,1])
#print(box.iloc[:2,2])
"""<2018.07.25>"""
"""데이터 갯수 세기"""
#10행의 데이터 생성
s=pd.Series(range(10))
#3번 인덱스에 NAN 생성
s[3]=np.nan
#print(s)
#count는 NAN의 개수를 세지 않는다.
#print("s의  NAN을 제외한 갯수는 {}".format(s.count()))
np.random.seed(2)
df=pd.DataFrame(np.random.randint(5,size=(4,4)),dtype=float)
df.iloc[2,3]=np.nan
#print(df)
#각 열마다 별도의 데이터 갯수를 세어주므로 데이터가 누락된 것을 찾을 수 있다.
#print(df.count())
"""연습 문제 1
다음 명령으로 타이타닉호 승객 데이터를 데이터프레임으로 읽어온다. 이 명령을 실행하려면 seaborn 패키지가 설치되어 있어야 한다.
import seaborn as sns
titanic = sns.load_dataset("titanic")
타이타닉호 승객 데이터의 데이터 값을 각 열마다 구해본다.
"""
import seaborn as sns
titanic=sns.load_dataset("titanic")
#print(titanic["age"].value_counts())
#print(titanic.head())
#print(titanic.count())
"""카테고리 값 세기"""
np.random.seed(1)
s2=pd.Series(np.random.randint(6,size=100))
#print(s2)
#tail()뒤에서 몇개만 보여준다
#print(s2.tail())
# 시리즈의 값이 정수, 문자열, 카테고리 값인 경우에 value_counts()는 값별로 몇개씩 존재하는지 알려준다.
#print(s2.value_counts())
"""정렬"""
#인덱스 기준 정렬
#print(s2.value_counts().sort_index())
#Value 기준 정렬
#print(s2.value_counts().sort_values())
#NaN값이 있는 경우에는 정렬하면 NAN값이 가장 나중에 나온다.
ran=pd.Series(range(10))
ran[8]=np.nan
#print(ran)
#print(ran.sort_values())
#큰 수에서 작은 수로 반대 정렬하려면 ascending=False로 지정
#print(ran.sort_values(ascending=False))
#sort_values메서드를 사용하려면 by인수로 정렬 기준이 되는 열을 지정할 수 있다.
#print(df.sort_values(by=1))
#print(df.sort_values(by=[1,2]))
"""
연습 문제 2
타이타닉호 승객중 성별(sex) 인원수, 나이별(age) 인원수, 선실별(class) 인원수, 사망/생존(alive) 인원수를 구하라.
"""
#print("Titanic의 탑승객 성별 구성은 {}".format(titanic["sex"].value_counts()))
#print("Titanic의 탑승객 연령별 구성은 {}".format(titanic["age"].value_counts().head()))
#print("Titanic의 선실별 인원 구성은 {}".format(titanic["class"].value_counts()))
#print("Titanic의 생존 인원수는 {}".format(titanic["alive"].value_counts()))
"""행/열 합계"""
#df2=pd.DataFrame(np.random.randint(10,size=(4,8)))
#print(df2)
##행별로 합계 구하기
#print(df2.sum(axis=1))
##열별로 합계 구하기
#print(df2.sum(axis=0))
#print(df2.sum())
#df2["RowSum"]=df2.sum(axis=1)
#print(df2)
#df2.loc["ColTotal",:]=df2.sum()
#print(df2)
"""apply변환"""
#행이나 열 단위로 더 복잡한 처리를 하고 싶을 때는 apply 메서드를 사용한다.
#인수로 행 또는 열 을 받는 함수를 apply 메서드의 인수로 넣으면 각 열(또는 행)을 반복하여 그 함수에 적용시킨다.
df3=pd.DataFrame({
        'A':[1,3,4,3,4],
        'B':[2,3,1,2,3],
        'C':[1,5,2,4,4]
        })
#print(df3)
#각 열의 최대값과 최소값의 차이를 구하고 싶으면 다음과 같은 람다 함수를 넣는다.
#print("각 열의 최대값과 최솟값의 차 \n{}".format(df3.apply(lambda x:x.max()-x.min())))
#만일 각 행에 대해서 적용하고 싶다면 axis=1의 인수를 사용한다.
#print("각 행의 최대값과 최솟값의 차 \n{}".format(df3.apply(lambda x:x.max()-x.min(),axis=1)))
#각 열에 대해 어떤값이 얼마나 사용되었는지 알고 싶다면 value_counts 함수를 넣을 수 있다.
#print(df3.apply(pd.value_counts))
#NaN값은 fillna 메서드를 사용하여 원하는 값으로 바꿀 수 있다.
#astype 메서드로 전체 데이터의 자료형을 바꾸는것도 가능하다.
#print(df3.apply(pd.value_counts).fillna(0).astype(int))
"""실수 값을 카테고리 값으로 변환(일정 범위에 데이터 넣기)"""
#cut:실수 값의 경계선을 지정하는 경우
#qcut:갯수가 똑같은 구간으로 나누는 경우
ages=[0,2,10,21,23,37,61,20,41,32,100]
bins=[1,15,25,35,60,99]
labels=["미성년자","청년","중년","장년","노년"]
cats=pd.cut(ages,bins,labels=labels)
#print(cats)
df4=pd.DataFrame(ages,columns=["ages"])
df4["age_cat"]=pd.cut(df4.ages,bins,labels=labels)
#print(df4)
#qcut 명령은 구간 경계선을 지정하지 않고 데이터 갯수가 같도록 지정한 수의 구간으로 나눈다.
#예를 들어 다음 코드는 1000개의 데이터를 4개의 구간으로 나누는데 각 구간은 250개씩의 데이터를 가진다.
data=np.random.randn(1000)
cats=pd.qcut(data,4,labels=["Q1","Q2","Q3","Q4"])
#print(cats)
#해당 데이터에 어떤 열이 있는 확인해보려고 ㅎㅎㅎ
#print(titanic.count())
#특정 조건을 가지는 행렬 추출하는 방법!!!
#old=titanic[titanic["age"]>40]
#print(old)
#산 사람과 죽은 사람 분류
alive=titanic[titanic["alive"]=='yes']
dye=titanic[titanic["alive"]=='no']
#산 사람과 죽은 사람 명수 확인
#print("alive:{} dye:{}".format(len(alive),len(dye)))
age_group=[1,19,30,40,60,99]
level=["미성년자","청년층","중년","장년","노년"]
alive_clf=pd.cut(alive["age"],age_group,labels=level)
#print("<타이타닉에 탄 승객 연령별 인원수>\n{}".format(titanic_clf))
titanic_clf=pd.cut(titanic["age"],age_group,labels=level)
#print("<타이나닉에서 산 사람들의 연령별 인원수>\n{}".format(alive_clf.value_counts()))
dye_clf=pd.cut(dye["age"],age_group,labels=level)
#print("<타이타닉에서 죽은 사람들의 연령별 인원수>\n{}".format(dye_clf.value_counts()))
titanic_rating=(titanic_clf.value_counts()/len(titanic_clf))*100
#print("<타이타닉에서 탑승한 사람들 연령별 비율>\n{}".format(titanic_rating))
alive_rating=(alive_clf.value_counts()/len(alive_clf))*100
#print("<타이타닉에서 산 사람들 연령별 비율>\n{}".format(alive_rating))
dye_rating=(dye_clf.value_counts()/len(dye_clf))*100
#print("<타이나틱에서 죽은 사람들 연령별 비율>\n{}".format(dye_rating))
"""인덱스 조작 건너뛰기"""
"""데이터프레임 병합"""
#첫번재 데이터 프레임
df1=pd.DataFrame({
        '고객번호':[1001,1002,1003,1004,1005,1006,1007],
        '이름':['경현','병관','규원','상규','윤정','동규','유진']}
        ,columns=['고객번호','이름']
)
#print(df1)
#두번째 데이터 프레임
df2=pd.DataFrame({
        '고객번호':[1001,1001,1005,1006,1008,1001],
        '금액':[10000,20000,15000,5000,10000,3000]}
        ,columns=['고객번호','금액']
)
#print(df2)
#merge 명령으로 위의 두 데이터프레임 df1, df2를 합치면 공통 열인 고객변호 열을 기주으로 데이터를 찾아서 합친다.
#이때 기본적으로는 양쪽 데이터프레임에 모두 키가 존재하는 데이터만 보여주는 inner join 방식을 사용한다.
mer_inn=pd.merge(df1,df2)
#print(mer_inn)
#outer join 방식은 키 값이 한쪽에만 있어도 데이터를 보여준다.
mer_out=pd.merge(df1,df2,how='outer')
#print(mer_out)
#left와 right join도 있다.
mer_left=pd.merge(df1,df2,how='left')
#print(mer_left)
mer_right=pd.merge(df1,df2,how='right')
#print(mer_right)
flw1=pd.DataFrame({
        '품종':['rose','rose','lily','lily'],
        '꽃잎너비':[0.4,0.3,0.5,0.3]},
        columns=['품종','꽃잎너비'])
#print(flw1)
flw2=pd.DataFrame({
        '품종':['rose','lily','lily','rose'],
        '꽃잎길이':[1.4,1.3,1.5,1.3]},
        columns=['품종','꽃잎길이'])
#print(flw2)
#만약 테이블에 키 값이 같은 데이터가 여러개 있는 경우에는 있을 수 있는 모든 경우의 수를 따져서 조합을 만들어 낸다.
garden_info=pd.merge(flw2,flw1)
#print(garden_info)
#열의 이름이 같은데 나타내는 정보가 다를 경우 on을 써서 기준열의 정보를 명시해야 한다.
consu1=pd.DataFrame({
        '고객명':['춘향','춘향','몽룡'],
        '날짜':['2018-01-01','2018-01-02','2018-01-01'],
        '데이터':['20000','30000','10000']})
consu2=pd.DataFrame({
        '고객명':['춘향','몽룡'],
        '데이터':['여자','남자'],
        })
#print(consu1,consu2)
consu_data=pd.merge(consu1,consu2,on='고객명')
#print(consu_data)
city1=pd.DataFrame({
        '도시':['서울','서울','서울','부산','부산'],
        '연도':[2000,2005,2010,2000,2005],
        '인구':[9853972,9762546,9631482,3655437,3512547]
        })
print(city1)
city2=pd.DataFrame(
        np.arange(12).reshape((6,2)),
        index=[['부산','부산','서울','서울','서울','서울'],
               [2000,2005,2000,2005,2010,2015]],
        columns=['데이터1','데이터2'])
print(city2)
city_info=pd.merge(city1,city2,left_on=['도시','연도'],right_index=True)
print(city_info)
Team1=pd.DataFrame(
        [[1.,2.],[3.,4.],[5.,6.]],
        index=['win','draw','lose'],
        columns=['Kor','US'])
print(Team1)
Team2=pd.DataFrame(
        [[7.,8.],[9.,10.],[11.,12.],[13.,14.]],
        index=['draw','lose','attack','defence'],
        columns=['UK','Jap'])
print(Team2)
Team=pd.merge(Team1,Team2,how='outer',left_index=True,right_index=True)
print(Team)
