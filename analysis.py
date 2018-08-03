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

# 배구 시즌 데이터 불러오기
Season_result=pd.read_csv('E:/대학교/졸업/졸업작품/웹크롤링/Webcrwaling and scraping using python/Season_result.csv',engine='python',encoding='EUC-KR')

#배구 시즌 데이터 인덱스 Date로 재설정하기
Season_result=Season_result.set_index("Date")

# Python에서 SQL문처럼 사용하는 방법 사이트 https://codeburst.io/how-to-rewrite-your-sql-queries-in-pandas-and-more-149d341fc53e

# 한 시즌에서 한국전력의 경기만 불러오기 테이블명.query(해당열조건 |(or) &(and) 해당열조건)
#print(Season_result.query("Home=='한국전력'|Away=='한국전력'"))

"""
# 2018.08.03 오늘은 1세트를 이겼을때 경기에서 이긴다는 개소리를 가지고 삽질을 해봤습니다 ㅎㅎㅎ.
KB=Season_result.query("Home=='KB손해보험'|Away=='KB손해보험'")
Jumbos=Season_result.query("Home=='대한항공'|Away=='대한항공'")
Blue=Season_result.query("Home=='삼성화재'|Away=='삼성화재'")
Wibee=Season_result.query("Home=='우리카드'|Away=='우리카드'")
OK=Season_result.query("Home=='OK저축은행'|Away=='OK저축은행'")
Vixtorm=Season_result.query("Home=='한국전력'|Away=='한국전력'")
Sky=Season_result.query("Home=='현대캐피탈'|Away=='현대캐피탈'")

SW=0
SkyHW=Sky.query("Home=='현대캐피탈'and result=='승'")
SkyAW=Sky.query("Away=='현대캐피탈'and result=='패'")
# 1세트를 이겼을때 승리한 횟수(Home)
for loop in range(0,len(SkyHW)):
    if(int(SkyHW["1st"][loop][0:2])>int(SkyHW["1st"][loop][3:5]))==True:
        SW+=1
# 1세트를 이겼을때 승리한 횟수(Away)
for loop in range(0,len(SkyAW)):
    if(int(SkyAW["1st"][loop][0:2])<int(SkyAW["1st"][loop][3:5]))==True:
        SW+=1
# 경기에서 1세트에 이겨서 경기를 승리한 확률
print("현대캐피탈이 %s 경기중 %s경기를 승리 / 1세트에서 승리해서 승리한 경기 : %s(%.4s)"%(len(Sky),len(SkyHW)+len(SkyAW),SW,float(SW/(len(SkyHW)+len(SkyAW)))))

BW=0
BlueHW=Blue.query("Home=='삼성화재'and result=='승'")
BlueAW=Blue.query("Away=='삼성화재'and result=='패'")
#print(JumbosHW)
# 1세트를 이겼을때 승리한 횟수(Home)
for loop in range(0,len(BlueHW)):
    if(int(BlueHW["1st"][loop][0:2])>int(BlueHW["1st"][loop][3:5]))==True:
        BW+=1
# 1세트를 이겼을때 승리한 횟수(Away)
for loop in range(0,len(BlueAW)):
    if(int(BlueAW["1st"][loop][0:2])<int(BlueAW["1st"][loop][3:5]))==True:
        BW+=1
# 경기에서 1세트에 이겨서 경기를 승리한 확률
print("삼성화재이 %s 경기중 %s경기를 승리 / 1세트에서 승리해서 승리한 경기 : %s(%.4s)"%(len(Blue),len(BlueHW)+len(BlueAW),BW,float(BW/(len(BlueHW)+len(BlueAW)))))

JW=0
JumbosHW=Jumbos.query("Home=='대한항공'and result=='승'")
JumbosAW=Jumbos.query("Away=='대한항공'and result=='패'")
#print(JumbosHW)
# 1세트를 이겼을때 승리한 횟수(Home)
for loop in range(0,len(JumbosHW)):
    if(int(JumbosHW["1st"][loop][0:2])>int(JumbosHW["1st"][loop][3:5]))==True:
        JW+=1
# 1세트를 이겼을때 승리한 횟수(Away)
for loop in range(0,len(JumbosAW)):
    if(int(JumbosAW["1st"][loop][0:2])<int(JumbosAW["1st"][loop][3:5]))==True:
        JW+=1
# 경기에서 1세트에 이겨서 경기를 승리한 확률
print("대한항공이 %s 경기중 %s경기를 승리 / 1세트에서 승리해서 승리한 경기 : %s(%.4s)"%(len(Jumbos),len(JumbosHW)+len(JumbosAW),JW,float(JW/(len(JumbosHW)+len(JumbosAW)))))

KW=0
KBHW=KB.query("Home=='KB손해보험'and result=='승'")
KBAW=KB.query("Away=='KB손해보험'and result=='패'")
# 1세트를 이겼을때 승리한 횟수(Home)
for loop in range(0,len(KBHW)):
    if(int(KBHW["1st"][loop][0:2])>int(KBHW["1st"][loop][3:5]))==True:
        KW+=1
# 1세트를 이겼을때 승리한 횟수(Away)
for loop in range(0,len(KBAW)):
    if(int(KBAW["1st"][loop][0:2])<int(KBAW["1st"][loop][3:5]))==True:
        KW+=1
# 경기에서 1세트에 이겨서 경기를 승리한 확률
print("KB손해보험이 %s 경기중 %s경기를 승리 / 1세트에서 승리해서 승리한 경기 : %s(%.4s)"%(len(KB),len(KBHW)+len(KBAW),VW,float(KW/(len(KBHW)+len(KBAW)))))

VW=0
VixtormHW=Vixtorm.query("Home=='한국전력'and result=='승'")
VixtormAW=Vixtorm.query("Away=='한국전력'and result=='패'")
# 1세트를 이겼을때 승리한 횟수(Home)
for loop in range(0,len(VixtormHW)):
    if(int(VixtormHW["1st"][loop][0:2])>int(VixtormHW["1st"][loop][3:5]))==True:
        VW+=1
# 1세트를 이겼을때 승리한 횟수(Away)
for loop in range(0,len(VixtormAW)):
    if(int(VixtormAW["1st"][loop][0:2])<int(VixtormAW["1st"][loop][3:5]))==True:
        VW+=1
# 경기에서 1세트에 이겨서 경기를 승리한 확률
print("한국전력이 %s 경기중 %s경기를 승리 / 1세트에서 승리해서 승리한 경기 : %s(%.4s)"%(len(Vixtorm),len(VixtormHW)+len(VixtormAW),VW,float(VW/(len(VixtormHW)+len(VixtormAW)))))

WW=0
WibeeHW=Wibee.query("Home=='우리카드'and result=='승'")
WibeeAW=Wibee.query("Away=='우리카드'and result=='패'")
# 1세트를 이겼을때 승리한 횟수(Home)
for loop in range(0,len(WibeeHW)):
    if(int(WibeeHW["1st"][loop][0:2])>int(WibeeHW["1st"][loop][3:5]))==True:
        WW+=1
# 1세트를 이겼을때 승리한 횟수(Away)
for loop in range(0,len(WibeeAW)):
    if(int(WibeeAW["1st"][loop][0:2])<int(WibeeAW["1st"][loop][3:5]))==True:
        WW+=1

# 경기에서 1세트에 이겨서 경기를 승리한 확률
print("우리은행이 %s 경기중 %s경기를 승리 / 1세트에서 승리해서 승리한 경기 : %s(%.4s)"%(len(Wibee),len(WibeeHW)+len(WibeeAW),WW,float(WW/(len(WibeeHW)+len(WibeeAW)))))

OW=0
OKHW=OK.query("Home=='OK저축은행'and result=='승'")
OKAW=OK.query("Away=='OK저축은행'and result=='패'")
# 1세트를 이겼을때 승리한 횟수(Home)
for loop in range(0,len(OKHW)):
    if(int(OKHW["1st"][loop][0:2])>int(OKHW["1st"][loop][3:5]))==True:
        OW+=1
# 1세트를 이겼을때 승리한 횟수(Away)
for loop in range(0,len(OKAW)):
    if(int(OKAW["1st"][loop][0:2])<int(OKAW["1st"][loop][3:5]))==True:
        OW+=1

# 경기에서 1세트에 이겨서 경기를 승리한 확률
print("OK저축은행이 %s 경기중 %s경기를 승리 / 1세트에서 승리해서 승리한 경기 : %s(%.4s)"%(len(OK),len(OKHW)+len(OKAW),OW,float(OW/(len(OKHW)+len(OKAW)))))
"""

"""
# 한국전력이 Home에서 승리한 횟수와 Away에서 승리한 횟수는 9 / 8 으로 별 차이는 없었다.
Kepco=Season_result.query("Home=='한국전력'|Away=='한국전력'")
#print(Kepco.query("Home=='한국전력' & result=='승'"))
Home=len(Kepco.query("Home=='한국전력'"))
Home_win=len(Kepco.query("Home=='한국전력' & result=='승'"))
Away=len(Kepco.query("Away=='한국전력'"))
#print(Kepco.query("Away=='한국전력' & result=='패'"))
Away_win=len(Kepco.query("Away=='한국전력' & result=='패'"))
print("한국전력의 총 경기 {} / Home 승리 {} / Home 패배 {} / Away 승리 {} / Away 패배 {}".format(len(Kepco),Home_win,Home-Home_win,Away_win,Away-Away_win) )
print("한국전력이 Home에서 이겼을 때 Set_score")
print(Kepco.query("Home=='한국전력' and result=='승'")["Set_score"].value_counts()) # 9개 데이터
print("한국전력이 Home에서 졌을 때 Set_score")
print(Kepco.query("Home=='한국전력' and result!='승'")["Set_score"].value_counts()) # 9개 데이터
print("한국전력이 Away에서 이겼을 때 Set_score")
print(Kepco.query("Away=='한국전력' and result=='패'")["Set_score"].value_counts()) 
print("한국전력이 Away에서 졌을 때 Set_score")
print(Kepco.query("Away=='한국전력' and result!='패'")["Set_score"].value_counts())

# =============================================================================================================

# 대한항공이 Home에서 승리한 횟수와 Away에서 승리한 횟수는 9 / 13 으로 별 차이는 없었다. Home에서보다 Away에서 더 잘했다???
Jumbos=Season_result.query("Home=='대한항공'|Away=='대한항공'")
#print(Jumbos.query("Home=='대한항공' & result=='승'"))
Home=len(Jumbos.query("Home=='대한항공'"))
Home_win=len(Jumbos.query("Home=='대한항공' & result=='승'"))
Away=len(Jumbos.query("Away=='대한항공'"))
#print(Jumbos.query("Away=='대한항공' & result=='패'"))
Away_win=len(Jumbos.query("Away=='대한항공' & result=='패'"))
print("대한항공의 총 경기 {} / Home 승리 {} / Home 패배 {} / Away 승리 {} / Away 패배 {}".format(len(Jumbos),Home_win,Home-Home_win,Away_win,Away-Away_win) )
print("대한항공이 Home에서 이겼을 때 Set_score")
print(Jumbos.query("Home=='대한항공' and result=='승'")["Set_score"].value_counts())
print("대한항공이 Home에서 졌을 때 Set_score")
print(Jumbos.query("Home=='대한항공' and result!='승'")["Set_score"].value_counts())
print("대한항공이 Away에서 이겼을 때 Set_score")
print(Jumbos.query("Away=='대한항공' and result=='패'")["Set_score"].value_counts()) 
print("대한항공이 Away에서 졌을 때 Set_score")
print(Jumbos.query("Away=='대한항공' and result!='패'")["Set_score"].value_counts())
"""

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

#store=pd.read_csv('fruit.csv',engine='python')
# Month를 인덱스롤 재설정
#store=store.set_index("Month")
#print(store)
#store.sum(axis=1).plot(kind="bar")
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