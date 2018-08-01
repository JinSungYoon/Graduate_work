# vollyball match data crawling file

import requests
import re
from bs4 import BeautifulSoup
import pandas as pd


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


"""
# 경기별 상세 기록 긁어오는 중....
html=requests.get('http://www.kovo.co.kr/game/v-league/11141_game-summary.asp?season=014&g_part=201&r_round=1&g_num=1&')

text=html.text

box = BeautifulSoup(text,'html.parser')

data = box.select('a')
#for loop in data:
#    print(loop.text)
#print(data)

#div>div[class=wrp_record wrp_precord]>table[class=lst_board lst_fixed w123]>tbody>td


#tab2 > div > div.wrp_lst > table.lst_board.lst_fixed.w123 > tbody > tr:nth-child(1) > td:nth-child(2) > a


#match.to_csv("match_data.csv",mode='w',encoding='EUC-KR')
"""