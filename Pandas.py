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
#인덱스 기반 연산
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
print(box.iloc[0,1])
print(box.iloc[:2,2])
"""데이터 갯수 세기"""
