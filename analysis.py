import csv
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

# 엑셀파일을 읽어옵니다.
#f=open('서울시 대중교통 수단별 이용 현황.csv','r',encoding='utf-8')
#rdr=csv.reader(f)
# Pandas로 데이터 읽어오기
# utf-8로 인코딩 된 파일 읽어오기
#table=pd.read_csv('서울시 대중교통 수단별 이용 현황.csv',delimiter=',',engine='python',encoding="utf-8")
# EUC-KR로 인코딩 된 파일 읽어오기
#test=pd.read_csv('서울교통공사 2016년 일별 역별 시간대별 승하차인원(1_8호선).csv',engine='python',encoding='EUC-KR')
# 데이터 프레임 만들기
"""
fruit=pd.DataFrame({
        '사과':np.random.randint(100,1000,size=10),
        '배':np.random.randint(100,1000,size=10),
        '참외':np.random.randint(100,1000,size=10),
        '옥수수':np.random.randint(100,1000,size=10),
        '고구마':np.random.randint(100,1000,size=10),
        '수박':np.random.randint(100,1000,size=10),
        '딸기':np.random.randint(100,1000,size=10),
        '토마토':np.random.randint(100,1000,size=10),
        },
        columns=['딸기','토마토','수박','참외','사과','배','옥수수','고구마'],
        index=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct']
        )
print(fruit)
fruit.index.name="Month"
# 열 데이터 추가
#fruit["sum"]=fruit.sum(axis=1)
print(len(fruit.columns))
print(len(fruit.index))
# 행 데이터 추가
fruit.loc["Nov"]=np.random.randint(100,1000,size=8)
fruit.loc["Dec"]=np.random.randint(100,1000,size=8)
print(fruit)
# 엑셀파일로 내보내기
fruit.to_csv("fruit.csv",mode='w',encoding='EUC-KR')
"""
store=pd.read_csv('fruit.csv',engine='python')
# Month를 인덱스롤 재설정
store=store.set_index("Month")
#print(store)
store.sum(axis=1).plot(kind="bar")
"""
# Data에 엑셀 내용 넣기
Data=[]
for line in rdr:
    Data.append(line)
# Pandas로 Dataframe에 넣기
#Seoul=pd.DataFrame(Data[1:len(Data)+1],columns=Data[0],index=np.arange(1,len(Data)))
#Seoul=pd.DataFrame(Data[3:len(Data)+1],columns=Data[0][3:13],index=Data[:][2:])
for loop in range(2,len(Data)):
    print(Data[loop][2:])
#print(Seoul.info())
#print(Seoul)
# iloc는 [행,열]을 적으면 해당 데이터를 긁어온다.
#move=Seoul.iloc[:,3:13].sum(axis=0)
move=[]
#print(Data[0][3:13])
month=Data[0][3:13]
# map(자료형,data)는 해당 자료를 입력 자료형 형태로 변환해주는 함수이다.
for index in range(1,len(Data)-1):
    move.append(list(map(int,Data[index][3:13])))
# Dataframe에서 강제로 문자열을 숫자로 바꾸는 함수
def coerce_df_columns_to_numeric(df, column_list):
    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
coerce_df_columns_to_numeric(Seoul,['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월'])
#월별 지하철 이용객 그래프로 표시
#Seoul.iloc[:,3:13].sum(axis=0).plot(kind="bar")
#Seoul["1월"].plot(kind="bar")
"""