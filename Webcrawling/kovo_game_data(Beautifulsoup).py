# vollyball match data crawling file
# reference
# <https://beomi.github.io/gb-crawling/posts/2017-01-20-HowToMakeWebCrawler.html>
# https://stackoverflow.com/questions/38028384/beautifulsoup-is-there-a-difference-between-find-and-select-python-3-x

import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
# 리스트에스 각 요소의 갯수 셀때 쓰는 import
from collections import Counter

# 5년치 남녀 시즌 순위 및 승점 승,패 세트 득실률 점수 득실률 크롤링
"""
MSeason = []
FSeason = []

# 시즌 부문별 순위도 긁어야 된다고 생각해서 일단 만들어 놓았지만 html에 데이터가 없다..... 어떻게 찾아야 할 지 잘 모르겠다....
MPart_table = [ [] for i in range(15)]

for num in range(8,15):
        # HTML데이터 긁어오기 - 부득이하게 하나의 페이지에 여자부 남자부 데이터가 없어서 남자 여자를 나눠야 했습니다.....
        
        # 남자부 시즌 경기 결과
        if num<10:
            MSeason_result_site = requests.get("http://www.kovo.co.kr/game/v-league/11210_team-ranking.asp?s_part=1&season=00{}&g_part=201".format(str(num)))
        else:
            MSeason_result_site = requests.get("http://www.kovo.co.kr/game/v-league/11210_team-ranking.asp?s_part=1&season=0{}&g_part=201".format(str(num)))        
    
        # 여자부 시즌 경기 결과
        if num<10:
            FSeason_result_site = requests.get("http://www.kovo.co.kr/game/v-league/11210_team-ranking.asp?season=00{}&g_part=201&s_part=2".format(str(num)))
        else:
            FSeason_result_site = requests.get("http://www.kovo.co.kr/game/v-league/11210_team-ranking.asp?season=0{}&g_part=201&s_part=2".format(str(num)))
        
        temp_site = requests.get("http://www.kovo.co.kr/game/v-league/11210_team-ranking.asp?s_part=1&season=014&g_part=201#tab1_5")        
        
        # Season_result를 text화 해서 html의 정보를 텍스트로 가져오기
        MSeason_text = MSeason_result_site.text
        FSeason_text = FSeason_result_site.text
        
        # 텍스트로 가져온 데이터를 BeautifulSoup의 형태로 변환
        MSeason_temp = BeautifulSoup(MSeason_text,'html.parser')
        FSeason_temp = BeautifulSoup(FSeason_text,'html.parser')
        
        # 시즌 결과 테이블만 Result_table에 저장
        MResult_table = MSeason_temp.select("table[class=lst_board]")[0].text
        FResult_table = FSeason_temp.select("table[class=lst_board]")[0].text
                
        Mtemp = MResult_table.splitlines()
        Ftemp = FResult_table.splitlines()
   
        # 데이터에서 불필요한 문자열 제거
        Mblank = Mtemp.count( '')
        Mtab = Mtemp.count('\t\t\t\t\t\t\t\t\t')
        Fblank = Ftemp.count( '')
        Ftab = Ftemp.count( '\t\t\t\t\t\t\t\t\t')
        
        for loop in range(Mblank):
            if loop<Mtab:
                Mtemp.remove( '')
                Mtemp.remove('\t\t\t\t\t\t\t\t\t')
            else:
                Mtemp.remove( '')
                
        for loop in range(Fblank):
            if loop<Ftab:
                Ftemp.remove( '')
                Ftemp.remove( '\t\t\t\t\t\t\t\t\t')
            else:
                Ftemp.remove( '')
        
       # 0번 : 테이블 설명 / 1번 : 순위 / 2번 : 팀 / 3번 :경기수 / 4번 : 승점 / 5번 : 승 / 6번 : 패 / 7번 : 세트득실률 / 8번 : 점수 득실률
        
        # 데이터 테이블 요소 생성
        MResult_list = []
        FResult_list = []
        
        # result_list에 columns값을 넣어놓는다.
        for i in range(1,9):
            MResult_list.append(Mtemp[i])
        
        MSeason_data = [[] for index in range(int(len(Mtemp)/9))]
        
        for i in range(1,9):
            FResult_list.append(Ftemp[i])
        
        FSeason_data = [[] for index in range(int(len(Ftemp)/9))]
        
        # Season_data의 인덱스 번호
        index=0
        for loop in range(9,len(Mtemp),8):
            MSeason_data[index].append(int(Mtemp[loop]))
            MSeason_data[index].append(Mtemp[loop+1])
            MSeason_data[index].append(int(Mtemp[loop+2]))
            MSeason_data[index].append(int(Mtemp[loop+3]))
            MSeason_data[index].append(int(Mtemp[loop+4]))
            MSeason_data[index].append(int(Mtemp[loop+5]))
            MSeason_data[index].append(float(Mtemp[loop+6]))
            MSeason_data[index].append(float(Mtemp[loop+7]))
            index+=1
        
        index = 0
        for loop in range(9,len(Ftemp),8):
            FSeason_data[index].append(int(Ftemp[loop]))
            FSeason_data[index].append(Ftemp[loop+1])
            FSeason_data[index].append(int(Ftemp[loop+2]))
            FSeason_data[index].append(int(Ftemp[loop+3]))
            FSeason_data[index].append(int(Ftemp[loop+4]))
            FSeason_data[index].append(int(Ftemp[loop+5]))
            FSeason_data[index].append(float(Ftemp[loop+6]))
            FSeason_data[index].append(float(Ftemp[loop+7]))
            index+=1

        
        # 경기 데이터를 Season_result라는 테이블로 만든다.
        MSeason_result = pd.DataFrame(MSeason_data,columns=MResult_list)
        FSeason_result = pd.DataFrame(FSeason_data,columns=FResult_list)
        
        # 테이블의 인덱스를 순위로 바꿔준다.
        MSeason_result=MSeason_result.set_index("팀")
        FSeason_result=FSeason_result.set_index("팀")
        MSeason.append(MSeason_result)
        FSeason.append(FSeason_result)
"""
# ====================================================한 시즌 경기 결과 데이터==============================================
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
result=[] # 12번 인덱스

for page in range(1,20):
    # KOVO HTTP GET Request
        kovo = requests.get('http://www.kovo.co.kr/stats/46101_previous-record.asp?s_part=1&spart=&t_code=&s_season=014&s_pr=201|1&e_season=014&e_pr=201|6&page={}'.format(str(page)))
        
        print("%s %s"%(page,kovo))
    
        # HTML text화 하기
        site = kovo.text
                
        # Beautifulsoup로 html 저장하기
        pot = BeautifulSoup(site,'html.parser')
        
        # 경기 관련 테이블에 있는 내용 다 긁었어유 ㅎㅎㅎ
        # soup.select("div[id=foo] > div > div > div[class=fee] > span > span > a") 쓰는 format
        
        # 내가 글어야 하는 데이터의 css 정보
        #print(pot.select("article[id=tab1] > div[class=wrp_lst] > table > tbody > tr > td[class=border_left]"))
        
        table=pot.select("article[id=tab1] > div[class=wrp_lst] > table > tbody > tr > td")
        
        # 해당 페이지를 초과했을때 빠져나올 수 있는 조건문
        if len(table)<10:
            break
        
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

print(match)

#match.to_csv("Season_result.csv",mode='w',encoding='EUC-KR')
"""
# 경기별 상세 기록 긁어오는 중....
#===========================선수들 이름 데이터 긁어오기===========================================
"""
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
"""
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
Htable=[[] for i in range(len(Hname))]
Atable=[[] for i in range(len(Aname))]

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
"""
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
"""
#================================경기 생중계 문자기록 데이터 크롤링=============================================
"""

Set_success=[]
Set_rate=[]

# 최종
# 한경기에 최소 3개에서 최대 5개의 경기가 있으므로 그 데이터를 다 긁어야 해서 5번 for 구문을 반복
for page in range(1,6):
    # 1세트 문자중계 데이터 크롤링 결과는 21:25 대한항공이 승리
    On_air=requests.get('http://www.kovo.co.kr/media/popup_result.asp?season=014&g_part=201&r_round=1&g_num=1&r_set=%s'%(page))
    
    # 가져온 데이터를 텍스트와
    On_text=On_air.text
    
    # 텍스트화한 정보를 BeautifulSoup로 처리
    hot=BeautifulSoup(On_text,'html.parser')
    
    # HTML 데이터에서 경기 생중계 문자 기록 데이터만 On_message에 저장
    On_message_text=hot.select("div[id=onair_lst]")[0].text
    
    # 경기 중계데이터 개행문자 기준으로 분할
    On_message_text=On_message_text.splitlines()
    
    # 필요없는 데이터 제거작업
    blank_num=On_message_text.count( '')
    sp_ch=On_message_text.count( '\xa0')
    
    for i in range(0,blank_num):
        if i<sp_ch:
            On_message_text.remove( '')
            On_message_text.remove( '\xa0')
        else:
            On_message_text.remove( '')
    
    # 각 단어 요소가 몇번씩 나왔는지 파악하는 함수
    Counter(On_message_text)
    
    # 현대 캐피탈 선수들의 이름을 Sky_name에 저장
    Sky_name=[]
    for loop in range(0,len(Hname)):
        Sky_name.append(Hname[loop][:-4])
    
    # 대한항공 선수들의 이름을 Jumbos_name에 저장
    Jumbos_name=[]
    for loop in range(0,len(Aname)):
        Jumbos_name.append(Aname[loop][:-4])
    
    Sky_data=[[] for i in range(len(Sky_name))]
    Jumbos_data=[[] for i in range(len(Jumbos_name))]
    
    # 현대캐피탈의 데이터만 긁어온 데이터
    for i in range(0,len(On_message_text)):         # 생중계 데이터의 길이만큼
        if len(On_message_text[i])>4:               # 팀득점, 팀실패, 경기포인트 등을 제외하고 출력하기 위해서
            for loop in range(len(Sky_name)):      # 현대캐피탈 선수들의 이름이 한번씩 들어가기 위해서
                if Sky_name[loop] in On_message_text[i]:   # 현대캐파틸 선수들의 이름이 들어간 텍스트 각 선수별로 Sky_data에 저장
                    Sky_data[loop].append(On_message_text[i])
    
    # 대한항공의 데이터만 긁어온 데이터
    for i in range(0,len(On_message_text)):         # 생중계 데이터의 길이만큼
        if len(On_message_text[i])>4:               # 팀득점, 팀실패, 경기포인트 등을 제외하고 출력하기 위해서
            for loop in range(len(Jumbos_name)):      # 대한항공 선수들의 이름이 한번씩 들어가기 위해서
                if Jumbos_name[loop] in On_message_text[i]:   # 대한항공 선수들의 이름이 들어간 텍스트 각 선수별로 Jumbos_data에 저장
                    Jumbos_data[loop].append(On_message_text[i])
    
    # 경기기록 종류 리스트
    Scoring_items=[
            '오픈','오픈 아웃','오픈 성공 득점',
            '시간차',
            '백어택','백어택 포히트','백어택 성공 득점',
            '속공','속공 포히트','속공 성공 득점',
            '퀵오픈 ','퀵오픈 성공 득점',
            '서브 ','서브 라인오버','서브 네트걸림',
            '스파이크서브 ','스파이크서브 아웃','스파이크서브 성공 득점','스파이크서브 네트걸림',
            '디그 ','디그 실패','디그 캐치볼','디그 기타범실',
            '세트 ','세트 성공','세트 오버네트',
            '리시브 ','리시브 실패','리시브 정확',
            '블로킹 ','블로킹 실패','블로킹 어시스트','블로킹 네트터치','유효블로킹 ','블로킹 성공 득점',
            '교체','투입'
            ]
    # 경기기록 대분류
    Scoring_sort=['오픈','시간차','백어택','속공','퀵오픈','서브','디그','세트','리시브','블로킹']
    
    # 팀별 성공,시도,성공률을 저장할 리스트 생성
    Sky_success=np.zeros(len(Scoring_sort))
    Jumbos_success=np.zeros(len(Scoring_sort))
    Sky_try=np.zeros(len(Scoring_sort))
    Jumbos_try=np.zeros(len(Scoring_sort))
    Sky_rate=np.zeros(len(Scoring_sort))
    Jumbos_rate=np.zeros(len(Scoring_sort))
    
    for index in range(0,len(Sky_data)):    # 현대캐피탈 생중계 메세지 경기 데이터
        for item in range(len(Scoring_sort)):     # 경기기록 종류 
            for record in range(len(Sky_data[index])):  # 선수별 데이터에서 돌도록
                    if Scoring_sort[item] in Sky_data[index][record]:   # 선수별 데이터에서 경기기록 종류가 있다면
                        if '성공' in Sky_data[index][record]:     # 그것이 '성공'을 포함하고 있다면
                            Sky_success[item]+=1
                        if '디그' in Sky_data[index][record]:     # 그것이 '디그'를 포함하고 있다면
                            Sky_success[item]+=1
                        if '정확' in Sky_data[index][record]:     # 그것이 '정확'을 포함하고 있다면
                            Sky_success[item]+=1
                        Sky_try[item]+=1
    
    for index in range(0,len(Jumbos_data)):     # 대한항공 생중계 메세지 경기 데이터
        for item in range(len(Scoring_sort)):     #경기기록 종류 
            for record in range(len(Jumbos_data[index])):   # 선수별 데이터에서 돌도록
                    if Scoring_sort[item] in Jumbos_data[index][record]:    # 선수별 데이터에서 경기기록 종류가 있다면
                        if '성공' in Jumbos_data[index][record]:  # 그것이 '성공'을 포함하고 있다면
                            Jumbos_success[item]+=1
                        if '디그' in Jumbos_data[index][record]:  # 그것이 '디그'를 포함하고 있다면
                            Jumbos_success[item]+=1
                        if '정확' in Jumbos_data[index][record]:  # 그것이 '정확'을 포함하고 있다면
                            Jumbos_success[item]+=1
                        Jumbos_try[item]+=1
    
    
    # 오픈과 퀵오픈이 오픈이라는 공통단어를 가지고 있으므로 오픈에서 퀵오픈의 개수를 빼줘야 한다.
    Sky_success[0]=Sky_success[0]-Sky_success[4]
    Jumbos_success[0]=Jumbos_success[0]-Jumbos_success[4]
    Sky_try[0]=Sky_try[0]-Sky_try[4]
    Jumbos_try[0]=Jumbos_try[0]-Jumbos_try[4]
    
    # 각 항목별 성공률을 계산해서 팀명_rate에 저장합니다.
    for loop in range(len(Scoring_sort)):
        Sky_rate[loop]=Sky_success[loop]/Sky_try[loop]
        Jumbos_rate[loop]=Jumbos_success[loop]/Jumbos_try[loop]
    
    # 세트별 각 항목의 성공률을 리스트화
    Set_rate.append(Sky_rate)
    Set_rate.append(Jumbos_rate)
    
    # 세트별 각 항목의 성공 갯수를 리스트화
    Set_success.append(Sky_success)
    Set_success.append(Jumbos_success)
    
#성공률 데이터 테이블화
Rate_record=pd.DataFrame(Set_rate,columns=Scoring_sort,index=["sky","Jumbos"]*5)
#성공횟수 데이터 테이블화
Success_record=pd.DataFrame(Set_success,columns=Scoring_sort,index=["sky","Jumbos"]*5)
"""