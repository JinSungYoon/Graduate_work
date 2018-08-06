# vollyball match data crawling file
# reference
# <https://beomi.github.io/gb-crawling/posts/2017-01-20-HowToMakeWebCrawler.html>
# https://stackoverflow.com/questions/38028384/beautifulsoup-is-there-a-difference-between-find-and-select-python-3-x

import requests
import re
from bs4 import BeautifulSoup
import pandas as pd

"""
# Create the table element

date=[] # 0번 인덱스
home=[] # 1번 인덱스
away=[] # 5번 인덱스
set_score=[]    # 6번 인덱스
First=[]  # 7번 인덱스
Second=[]  # 8번 인덱스
Third=[]  # 9번 인덱스
Fourth=[]  # 10번 인덱스
Fifth=[]  # 11번 인덱스
result=[] # 12번 인덱스

# Crawling the vollye_match_result9

for day in range(1,14):
    # KOVO HTTP GET Request
    kovo = requests.get('http://www.kovo.co.kr/stats/46101_previous-record.asp?s_part=1&spart=&t_code=&s_season=014&s_pr=201|1&e_season=014&e_pr=201|6&page={}'.format(str(day)))
    
    # HTML text화 하기
    site = kovo.text
    
    # Beautifulsoup로 html 저장하기
    pot = BeautifulSoup(site,'html.parser')
    
    # 경기 관련 테이블에 있는 내용 다 긁었어유 ㅎㅎㅎ
    # soup.select("div[id=foo] > div > div > div[class=fee] > span > span > a") 쓰는 format
    
    # 내가 글어야 하는 데이터의 css 정보
    #print(pot.select("article[id=tab1] > div[class=wrp_lst] > table > tbody > tr > td[class=border_left]"))
    
    table=pot.select("article[id=tab1] > div[class=wrp_lst] > table > tbody > tr > td")
        
    for loop in range(0,len(table),13):
        date.append(table[loop].text)
        home.append(table[loop+1].text)
        away.append(table[loop+5].text)
        set_score.append(table[loop+6].text)
        First.append(table[loop+7].text)
        Second.append(table[loop+8].text)
        Third.append(table[loop+9].text)
        Fourth.append(table[loop+10].text)
        Fifth.append(table[loop+11].text)
        result.append(table[loop+12].text)

match = pd.DataFrame({
            'Home':home,
            'Away':away,
            'Set_score':set_score,
            '1st':First,
            '2st':Second,
            '3st':Third,
            '4st':Fourth,        
            '5st':Fifth,
            'result':result
            },
            columns=["Home","Away","Set_score","1st","2st","3st","4st","5st","result"],
            index=date
        )       
match.index.name="Date"
#print(match)
match.to_csv("Season_result.csv",mode='w',encoding='EUC-KR')
"""

# 경기별 상세 기록 긁어오는 중....
#===========================선수들 이름 데이터 긁어오기===========================================

# 경기 상세 기록 데이터가 있는 사이트 HTML데이터 긁어오기
game_data=requests.get('http://www.kovo.co.kr/game/v-league/11141_game-summary.asp?season=014&g_part=201&r_round=1&g_num=1&')

# 경기 상세 기록 데이터 사이트 HTML정보 text화
text=game_data.text

# text데이터 BeautifulSoup로 넣음
dish=BeautifulSoup(text,'html.parser')

#print(len(dish.select('table')))
#print(len(dish.find_all('table')))

#5번은 현대캐피탈 선수 리스트
#6번은 경기기록 1번 테이블
#7번은 경기기록 2번 테이블
#8번은 경기기록 3번 테이블
#9번은 경기기록 4번 테이블

#10번은 대한항공 선수 리스트
#11번은 경기기록 1번 테이블
#12번은 경기기록 2번 테이블
#13번은 경기기록 3번 테이블
#14번은 경긱기록 4번 테이블

# 이름 정리 완료!!!
# name에 이름이 있는 태그 데이터 텍스트화해서 넣기
Home_name=dish.select('table')[5].text
Away_name=dish.select('table')[10].text

# 공백 제거
Home_name=Home_name.strip()
Away_name=Away_name.strip()

# 개행문자 기준으로 나눠서 리스트화
Home_name=Home_name.splitlines()
Away_name=Away_name.splitlines()

#  ''라는 쓰래기 문자들이 있어서 제거하는 과정
Home_num=Home_name.count( '')
Away_num=Away_name.count( '')
#print(Home_num)
#print(Away_num)

# 문자 아닌것만 리스트에 다시 정리
for loop in range(0,Home_num):
    Home_name.remove( '')
for loop in range(0,Away_num):
    Away_name.remove( '')

#print(Home_name)
#print(Away_name)

Hback_num=[]
Hname=[]
Aback_num=[]
Aname=[]

for loop in range(3,len(Home_name)):
    if loop%2==1:
        Hback_num.append(Home_name[loop])
    else:
        Hname.append(Home_name[loop])
        
for loop in range(3,len(Away_name)):
    if loop%2==1:
        Aback_num.append(Away_name[loop])
    else:
        Aname.append(Away_name[loop])        

#print(Hback_num)    
#print(Hname)

#print(Aback_num)
#print(Aname)

#==================================1번 차트 정리(경기 세부데이텨)=============================================
"""        


#1번 차트
Hchart_1=dish.select('table')[6].text
Achart_1=dish.select('table')[11].text    

# 띄어쓰기 없애기
Hchart_1=Hchart_1.strip()
Achart_1=Achart_1.strip()

# 개행단위로 나눠서 리스트화 하기
Hchart_1=Hchart_1.splitlines()
Achart_1=Achart_1.splitlines()

Hgar=Hchart_1.count( '')
Agar=Achart_1.count( '')

for loop in range(0,Hgar):
    Hchart_1.remove( '')
for loop in range(0,Agar):
    Achart_1.remove( '')

# 데이터가 들어갈 테이블 생성
Htable=[[] for i in range(14)]
Atable=[[] for i in range(12)]

#print(len(Hchart_1))
# 총계 : 289 / 목차 23개 / 선수 14명 * 항목 18개 = 252
#print(len(Achart_1))
# 총계 : 251 / 목차 23개 / 선수 12명 * 항목 18개 = 216

# 각 팀의 목차
Hindex=Hchart_1[5:23]
Hindex.insert(0,Hchart_1[1])
Aindex=Achart_1[5:23]
Aindex.insert(0,Achart_1[1])

# 목차는 제외하고 넣어야 하기 때문에
point=23
# Home팀의 경기 데이터를 Htable에 넣는다
for num in range(14):
    if num!=0:
        point+=19
    for index in range(19):
            Htable[num].append(Hchart_1[point+index])
point=23
# Away팀의 경기 데이터를 Atable에 넣는다.
for num in range(12):
    if num!=0:
        point+=19
    for index in range(19):
            Atable[num].append(Achart_1[point+index])

#print(Htable)            
#print(Atable)

# 경기데이터와 인덱스를 합쳐서 데이터 프레임을 만든다.
Hframe=pd.DataFrame(Htable,columns=Hindex)
Aframe=pd.DataFrame(Atable,columns=Aindex)

#print(Hframe)
#print(Aframe)
"""
#=========================================1~4번 테이블 데이터=======================================

# 데이터가 들어갈 테이블 생성
Htable=[[] for i in range(14)]
Atable=[[] for i in range(12)]
# 홈경기의 작은 columns
HSindex=[]
# 어웨이경기의 작은 columns
ASindex=[]
# 홈경기의 큰 columns
HBindex=[]
# 어웨이경기의 큰 columns
ABindex=[]

# 인덱스 임시 저장소
HTindex=[[],[]]
ATindex=[[],[]]

# 전체 다중 columns
Hindex=[]
Aindex=[]

for page in range(6,10):
    Hchart_4=dish.select('table')[page].text
    Achart_4=dish.select('table')[page+5].text    
    
    # 띄어쓰기 없애기
    Hchart_4=Hchart_4.strip()
    Achart_4=Achart_4.strip()
    
    # 개행단위로 나눠서 리스트화 하기
    Hchart_4=Hchart_4.splitlines()
    Achart_4=Achart_4.splitlines()
    
    Hgar=Hchart_4.count( '')
    Agar=Achart_4.count( '')
    
    for loop in range(0,Hgar):
        Hchart_4.remove( '')
    for loop in range(0,Agar):
        Achart_4.remove( '')
    
    #print(len(Hchart_4))
    # 1페이지
    # 목차 23개 / 선수 14/12명 * 항목 19개 
    # 2페이지
    # 목차 22개 / 선수 14/12명 * 항목 18개
    # 3페이지
    # 목차 21개 / 선수 14/12명 * 항목 17개
    # 4페이지
    # 목차 24개 / 선수 14/12명 * 항목 20개

#    1번 테이블 항목(19) / 2번 테이블 항목(18) / 3번 테이블 항목(17) / 4번 테이블 항목(20)
    if page==6:
#        print(Hchart_4[1:23])
         item=len(Hchart_4[1:23])  # 22
    elif page==7:
#        print(Hchart_4[1:22])
         item=len(Hchart_4[1:22])   # 21
    elif page==8:
#        print(Hchart_4[1:21])
         item=len(Hchart_4[1:21])   # 20
    elif page==9:
#        print(Hchart_4[1:24])
         item=len(Hchart_4[1:24])   # 23
    
    # 각 팀의 목차
    if page==6:
        # 득점은 하나의 카테고리이기 때문에 넣어준다.
        HTindex[0].append(Hchart_4[1])
        ATindex[0].append(Achart_4[1])
        # 공격종합 오픈 시간차는 6개의 항목이다.
        for i in range(2,5):
            for loop in range(0,6):
                HTindex[0].append(Hchart_4[i])
                ATindex[0].append(Achart_4[i])
        hindex=Hchart_4[5:23]
        hindex.insert(0,Hchart_4[1])
        aindex=Achart_4[5:23]
        aindex.insert(0,Achart_4[1])
    elif page==7:
        # 이동 후위 속공은 6개의 항목이다.
        for i in range(1,4):
            for loop in range(0,6):
                HTindex[0].append(Hchart_4[i])
                ATindex[0].append(Hchart_4[i])
        hindex=Hchart_4[4:22]
        aindex=Achart_4[4:22]
    elif page==8:
        for i in range(1,4):
            # 서브의 해당 항목이 5개 이므로
            if i==2:   
                for loop in range(0,5):
                    HTindex[0].append(Hchart_4[i])
                    ATindex[0].append(Hchart_4[i])
            #퀵 오픈, 디그의 항목이 6개 이므로
            else:
                for loop in range(0,6):
                    HTindex[0].append(Hchart_4[i])
                    ATindex[0].append(Hchart_4[i])
        hindex=Hchart_4[4:21]
        aindex=Achart_4[4:21]
    elif page==9:
        for i in range(1,4):
            # 블로킹은 해당 항목이 8개이므로
            if i==3:
                for loop in range(0,8):
                    HTindex[0].append(Hchart_4[i])
                    ATindex[0].append(Hchart_4[i])
            # 세트와 리시브는 해당 항목이 5개 이므로
            else:
                for loop in range(0,5):
                    HTindex[0].append(Hchart_4[i])
                    ATindex[0].append(Hchart_4[i])
        HTindex[0].append(Hchart_4[4])
        HTindex[0].append(Hchart_4[5])
        ATindex[0].append(Hchart_4[4])
        ATindex[0].append(Hchart_4[5])
        # 안타깝게도 벌칙 범실이 앞에 나와있어서 다시 뒤로 보내줘야 합니다.......
        hindex=Hchart_4[6:24]
        hindex.append(Hchart_4[4])
        hindex.append(Hchart_4[5])
        aindex=Achart_4[6:24]
        aindex.append(Achart_4[4])
        aindex.append(Achart_4[5])
               

    # 목차는 제외하고 넣어야 하기 때문에
    point=len(hindex)+4
    
    # Home팀의 경기 데이터를 Htable에 넣는다
    for num in range(14):
        if num!=0:
            point+=item-3
        for index in range(item-3):
                Htable[num].append(float(Hchart_4[point+index]))
    point=len(aindex)+4
    # Away팀의 경기 데이터를 Atable에 넣는다.
    for num in range(12):
        if num!=0:
            point+=item-3
        for index in range(item-3):
                Atable[num].append(float(Achart_4[point+index]))

    # 인덱스들도 하나로 합쳐야 한다.
    for loop in range(0,len(hindex)):
        HTindex[1].append(hindex[loop])
        ATindex[1].append(aindex[loop])

    Hindex=pd.MultiIndex.from_arrays(HTindex)    
    Aindex=pd.MultiIndex.from_arrays(ATindex)    
    
    
    # 경기데이터와 인덱스를 합쳐서 데이터 프레임을 만든다.
    Hframe=pd.DataFrame(Htable,columns=Hindex,index=Hname)
    Aframe=pd.DataFrame(Atable,columns=Aindex,index=Aname)

#Hframe.to_csv("H_data.csv",mode='w',encoding='EUC-KR')
#Aframe.to_csv("A_data.csv",mode='w',encoding='EUC-KR')

#=============================================================================
"""
# 경기 문자기록 데이터 크롤링
# 참고로 세트별로 다 있다 ㅎㅎㅎ 어떻게 긁을래 ㅋㅋㅋㅋ???
message=requests.get('http://www.kovo.co.kr/media/popup_result.asp?season=014&g_part=201&r_round=1&g_num=1')

m_text=message.text

mashroom=BeautifulSoup(m_text,'html.parser')

# 경기 실시간 문자중계 데이터 텍스트와
M_data=mashroom.select("div[id=onair_lst]")[0].text

# 개행문자 중심으로 분할
onair_data=M_data.splitlines()
# 띄어쓰기 문자들 갯수 파악
blank_num=onair_data.count( '')
# 이상한 문자 갯수 파악
sp_ch=onair_data.count( '\xa0')
# 띄어쓰기 문자들 제거
for loop in range(0,blank_num):
   if loop<526:
    onair_data.remove( '')
    onair_data.remove( '\xa0')
   else:
       onair_data.remove( '')
print(onair_data)
"""