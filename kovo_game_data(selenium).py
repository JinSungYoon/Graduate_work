from selenium import webdriver as wd
import pandas as pd
import numpy as np

# Chrome의 경우 다운 받은 chromedriver의 위치를 지정해 준다.
driver = wd.Chrome(executable_path='E:/대학교/졸업/졸업작품/웹크롤링/Web_Crawler(SKtechx Tacademy)/Chromedriver.exe')

# 암묵적으로 웹 자원 로드를 위해 3초를 기다려 준다.
driver.implicitly_wait(3)

# Kovo 사이트 url에 접근한다.
driver.get("http://www.kovo.co.kr/main.asp")

# 참조 사이트 : http://selenium-python.readthedocs.io/locating-elements.html

# 상단의 Game -> V-리그 -> 팀 순위로 접근한다.
driver.find_element_by_class_name('t_nav1').click()

driver.find_element_by_link_text('V-리그').click()

driver.find_element_by_link_text('팀 순위').click()

# 팀 순위 페이지에서 팀 순위 테이블을 긁어온다.
Result_table_text = driver.find_elements_by_css_selector('tbody')[0].text
Result_list_text = driver.find_elements_by_css_selector('thead')[0].text

# 팀 순위 페이지에서 부문별 순위 테이블을 긁어온다.
Part_table_text = driver.find_elements_by_css_selector('tbody')[1].text
Part_list_text = driver.find_elements_by_css_selector('thead')[1].text

# 팀 순위 테이블의 텍스트를 띄어쓰기 기준으로 나눈다.
Result_table_text = Result_table_text.splitlines()
Result_list_text = Result_list_text.split(' ')

# 부문별 테이블의 특스트를 띄어쓰기 기준으로 나눈다.
Part_table_text = Part_table_text.splitlines()
Part_list_text = Part_list_text.split(' ')

for loop in range(len(Result_table_text)):
    Result_table_text[loop] = Result_table_text[loop].split(' ')

for loop in range(len(Part_table_text)):
    Part_table_text[loop] = Part_table_text[loop].split(' ')

# 각 테이블의 숫자 데이터를 float와 int형으로 변경해 준다.
for loop in range(len(Result_table_text)):
    for index in range(len(Result_table_text[loop])):
        if index!=1:
            if index>5:
                Result_table_text[loop][index] = float(Result_table_text[loop][index])
            else:
                Result_table_text[loop][index] = int(Result_table_text[loop][index])

for loop in range(len(Part_table_text)):
    for index in range(len(Part_table_text[loop])):
        if index!=1:
            Part_table_text[loop][index] = int(Part_table_text[loop][index])

#print(Result_table_text)    
#print(Part_table_text)    

# 팀 순위 및 득점 부분 테이블 완성
Result_table = pd.DataFrame(Result_table_text,columns = Result_list_text)
Part_table = pd.DataFrame(Part_table_text,columns = Part_list_text)

# 인덱스를 팀으로 변경
Result_table = Result_table.set_index("팀")
Part_table = Part_table.set_index("팀")

#print(Result_table)
#print(Score_table)

Part_list = ['득점','공격','오픈공격','시간차공격','이동공격','후위공격','속공','퀵오픈','서브','블로킹','디그','세트','리시브','벌칙','범실']

#===========================================================부문별 파트 접근해서 데이터 긁어오기 =================================================
# 득접 부분은 메인페이지 긁을때 들어가서 0번 인덱스가 아닌 1번 인덱스부터 시작하는 것입니다.
for index in range(1,len(Part_list)):

    # 암묵적으로 웹 자원 로드를 위해 10초를 기다려 준다.(기다리지 않으면 제대로 로드 되지 않아 데이터가 없다는 에러가 자주 뜬다...)
    driver.implicitly_wait(10)
            
    # 항목의 파트 클릭
    item = driver.find_element_by_link_text(Part_list[index])
    item.click()
    
    # 공격 파트에 table의 text 접근
    item_table_mass = driver.find_element_by_id('tab1_%s'%(index+1)).text
    
    # text를 줄 바꿈 기준으로 분할
    item_table_mass = item_table_mass.splitlines()

#    인덱스 에러떠서 임시적으로 해놓은 코든이니 문제 발생이 생기지 않으면 살포시 지우세요 ㅎㅎㅎ    
#    print(len(item_table_mass))
    
    # 테이블의 columns값을 따로 추출
    item_list_text = item_table_mass[0] 
    
    # 테이블의 data를 따로 추출
    item_table_text = item_table_mass[1:]
    
    # 띄어쓰기 기준으로 모든 요소들을 분할
    for loop in range(0,len(item_table_text)):
        item_table_text[loop] = item_table_text[loop].split(' ')
    
    item_list_text = item_list_text.split(' ')
    
    item_columns = [[],[]]
    
    for i in range(len(item_list_text)):
        item_columns[0].append(item.text)
    for i in range(len(item_list_text)):
        item_columns[1].append(item_list_text[i])
    
    item_list = pd.MultiIndex.from_arrays(item_columns)
    
    # 득점 ~ 퀵오픈까지는 성공률에 %가 붙지만 그 이후에는 붙지 않는다.
    if index<8:
        # 테이블의 숫자들을 int 혹은 float 요소로 변경
        for loop in range(len(item_table_text)):
            for index in range(len(item_table_text[loop])):
                # 1번 인덱스는 팀명이므로
                if index!=1:
                    # 마지막 인덱스는 %가 붙어있어서 그거를 제외하고 float형으로 변환
                    if index==(len(item_table_text[loop])-1):
                        item_table_text[loop][index] = float(item_table_text[loop][index][:-1])
                    # 나머지 인덱스는 int형으로 변환
                    else:
                        item_table_text[loop][index] = int(item_table_text[loop][index])
    else:
        # 테이블의 숫자들을 int 혹은 float 요소로 변경
        for loop in range(len(item_table_text)):
            for index in range(len(item_table_text[loop])):
                # 1번 인덱스는 팀명이므로
                if index!=1:
                    # 마지막 인덱스는 %가 붙어있어서 그거를 제외하고 float형으로 변환
                    if index==(len(item_table_text[loop])-1):
                        item_table_text[loop][index] = float(item_table_text[loop][index])
                    # 나머지 인덱스는 int형으로 변환
                    else:
                        item_table_text[loop][index] = int(item_table_text[loop][index])
                    
    # 공격에 대한 테이블 형성
    item_table = pd.DataFrame(item_table_text,columns=item_list)
    
    # 테이블의 인덱스를 팀으로 설정
    item_table = item_table.set_index(item_table[item.text]["팀"])
    
    #print(Attack_table)
    
    # 참고 : https://datascienceschool.net/view-notebook/7002e92653434bc88c8c026c3449d27b/
    #        http://nittaku.tistory.com/121
    
    Part_table = pd.merge(Part_table,item_table,left_index=True,right_index=True)
    
print(Part_table)

Result_table.to_pickle('Kovo_result_table')
Part_table.to_pickle('Kovo_part_table')
