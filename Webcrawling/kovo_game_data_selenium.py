from selenium import webdriver as wd
import pandas as pd
import numpy as np

# Chrome의 경우 다운 받은 chromedriver의 위치를 지정해 준다.
driver = wd.Chrome(executable_path='E:/대학교/졸업/졸업작품/웹크롤링/Web_Crawler(SKtechx Tacademy)/Chromedriver.exe')

# 암묵적으로 웹 자원 로드를 위해 3초를 기다려 준다.
driver.implicitly_wait(3)
"""
# Kovo 사이트 url에 접근한다.
driver.get("http://www.kovo.co.kr/main.asp")

# 참조 사이트 : http://selenium-python.readthedocs.io/locating-elements.html

# 상단의 Game -> V-리그 -> 팀 순위로 접근한다.
driver.find_element_by_class_name('t_nav1').click()

driver.find_element_by_link_text('V-리그').click()

driver.find_element_by_link_text('팀 순위').click()

# find_element_by_class_name이랑 css.selector로 안 되서 xpath로 했음
driver.find_element_by_xpath("//a[@class='selectBox selectbox_custom w228 selectBox-dropdown']").click()

driver.find_element_by_link_text('NH농협 2011-2012 V-리그').click()

# 여자부를 클릭한다
#driver.find_element_by_link_text('여자부').click()

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

Part_index = [[],[]]

# 부문별 카테고리의 제목들이다.
Part_list = ['득점','공격','오픈공격','시간차공격','이동공격','후위공격','속공','퀵오픈','서브','블로킹','디그','세트','리시브','벌칙','범실']

# Part_index에 득점과 다른 인덱스를 넣어서 멀티 인덱스 준비를 한다.
for loop in range(len(Part_list_text)):
    Part_index[0].append(Part_list[0])
    Part_index[1].append(Part_list_text[loop])

# 다중 인덱스 형성
Part_index = pd.MultiIndex.from_arrays(Part_index)
#print(Part_index)

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
Part_table = pd.DataFrame(Part_table_text,columns = Part_index)

# 인덱스를 팀으로 변경
Result_table = Result_table.set_index("팀")
Part_table = Part_table.set_index(('득점','팀'))

# Result_table로 병합한다.
Result_table = pd.merge(Result_table,Part_table,left_index=True,right_index=True)

#print(Result_table)
#print(Score_table)

#===========================================================부문별 파트 접근해서 데이터 긁어오기 =================================================
# 득접 부분은 메인페이지 긁을때 들어가서 0번 인덱스가 아닌 1번 인덱스부터 시작하는 것입니다.
for index in range(1,len(Part_list)):

    # 항목의 파트 클릭
    item = driver.find_element_by_link_text(Part_list[index])
    item.click()

    # 암묵적으로 웹 자원 로드를 위해 1000초를 기다려 준다.(기다리지 않으면 제대로 로드 되지 않아 데이터가 없다는 에러가 자주 뜬다...)
    driver.implicitly_wait(1000)
                
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
    
    Result_table = pd.merge(Result_table,item_table,left_index=True,right_index=True)
    
Result_table.to_pickle('Kovo_Male_result_table(11-12)')
#Result_table.to_pickle('Kovo_Female_result_table(11-12)')
#Part_table.to_pickle('Kovo_part_table')
"""

# =================================================== 최대 연속승 최대 연속패를 추출하기 위한 역대 전적 크롤링 ======================================================================================

# Kovo 사이트 url에 접근한다.
driver.get("http://www.kovo.co.kr/main.asp")

# Stats를 접근한다.
driver.find_element_by_class_name('t_nav4').click()

# 역대기록에 접근한다.
driver.find_element_by_link_text('역대기록').click()

# 여자부를 클릭한다
#driver.find_element_by_link_text('여자부').click()

num = 12

for year in range(num,10,-1):

    if year != 17:
        # 16년도 시즌 이전에는
        driver.find_elements_by_xpath("//a[@class='selectBox selectbox_custom w286 selectBox-dropdown']")[0].click()
    
        driver.find_elements_by_link_text("NH농협 20%s-20%s V-리그"%(str(year),str(year+1)))[0].click()
        
        driver.implicitly_wait(10)
        
        driver.find_elements_by_xpath("//a[@class='selectBox selectbox_custom w286 selectBox-dropdown']")[1].click()

        driver.find_elements_by_link_text("NH농협 20%s-20%s V-리그"%(str(year),str(year+1)))[1].click()   
    
    # 잠시 30초 쉰다
    driver.implicitly_wait(10)
    
    # 라운드를 바꾸는 두번째 select-box를 클릭한다.
    driver.find_elements_by_xpath("//a[@class='selectBox selectbox_custom w123 selectBox-dropdown']")[1].click()
        
    # 6라운드를 선택한다.(만일 6라운드가 없을시 5라운드를 선택한다.....13-14년도가 5라운드라서...... )
    try :
        driver.find_element_by_link_text('6 Round').click()
    except:
        driver.find_element_by_link_text('5 Round').click()
        
    # 기록보기를 클릭한다.
    driver.find_element_by_link_text('기록보기').click()
    
    # 테이블의 columns을 지정합니다.
    table_columns = ['경기일자','홈','상대팀', '세트스코어', '1', '2', '3', '4', '5', '승패']
    table = '\n'
    # 페이지 이동
    for page in range(1,20):
        try:
            if page % 10 != 1:
                driver.find_element_by_link_text('{}'.format(str(page))).click()    
            print("page:%s"%(page))
    #        print(driver.find_elements_by_css_selector('tbody')[0].text)
    #        table.append(driver.find_elements_by_css_selector('tbody')[0].text)
            table = table +'\n' + driver.find_elements_by_css_selector('tbody')[0].text
            print("=============================================================================")
            if page % 10 == 0:
                driver.find_element_by_link_text('다음페이지로이동').click()
        except:
            break
    
    # 개행 단위로 나누고 띄어쓰기 단위로 나누기
    table = table.splitlines()
    for loop in range(len(table)):
        table[loop] = table[loop].split(' ')
        # 중간에 필요없는 내용을 제거한다.
        del table[loop][2:7]    
    # 앞에 개행단위 때문에 생성된 리스트를 제거한다.
    del table[0:2]
    
    # 시즌 데이터를 테이블로 만듭니다.
    Season_result = pd.DataFrame(table,columns=table_columns)
    # 시즌 데이터를 pickle 파일로 만든다.
    Season_result.to_pickle('Female_season(%s-%s)'%(year,year+1))