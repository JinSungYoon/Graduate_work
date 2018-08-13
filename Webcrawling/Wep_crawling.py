"""
<SKsechx Tacademy>
Python으로 웹 크롤러 만들기 Youtube
https://www.youtube.com/watch?v=TWb4xTwR0I8&list=PL9mhQYIlKEhf0DKhE-E59fR-iu7Vfpife
"""

#인터파크 투어 사이트에서 여행지를 입력 후 검색 --> 잠시후 --> 경과
#로그인시 PC 웹 사이트에서 처리가 어려울 경우 --> 모파일 로그인으로 진입 

#모듈 가져오기
#pip install selenium
#from selenium import webdriver as driver
import time

from selenium import webdriver as wd

from selenium.webdriver.common.by import By
# 명시적 대기를 위해서
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from Tour import TourInfo


# 사전에 필요한 정보를 로드 --> 디비 혹은 쉘, 배치 파일에서 인자로 받아서 세팅
main_url = 'http://air.interpark.com/'
test_url='http://www.naver.com/'
kovo_url = 'http://www.kovo.co.kr/'
keyword = '로마'
enter = '영화'
#상품 정보를 담는 리스트 (TourInfo 리스트)
tour_list = []

# 드라이버 로드
driver = wd.Chrome(executable_path='Chromedriver.exe')
# 차후 --> 옵셕 부여하여(프록시, 에이전트 조작, 이미지를 배제 )
# 크롤링을 오래 돌리면 --> 임시파일들이 쌓인다!! --> Temp파일 삭제

# 사이트 접속 (get)
driver.get(main_url)
#driver.get(test_url)
#driver.get(kovo_url)

# 검색창을 찾아서 검색어를 입력
# id : SearchGNBText
driver.find_element_by_id('SearchGNBText').send_keys(keyword)
# 네이버 검색창 id : query
#driver.find_element_by_id('query').send_keys(enter)

# 수정할 경우 --> 뒤에 내용이 붙어버림 --> .clear() --> .send_keys('내용')

# 검색버튼을 클릭
# 버튼의 id가 없으므로 버튼의 클래스 명을 넣는다.
driver.find_element_by_css_selector('button.search-btn').click()
# 네이버는 id가 있으므로 id로 접근
#driver.find_element_by_id('search_btn').click()
# 배구영맹 STATS를 클릭해야 한다
#driver.find_element_by_class_name('t_nav4').click()

# 잠시 대기 --> 페이지가 로드되고 나서 즉각적으로 데이터를 획득하는 행외는 자제->
# 명시적 대기 --> 특정 요소가 로케이트(발견될때까지) 대기
try:
    element = WebDriverWait(driver,10).until(
            # 지정한 한개 요소가 올라오면 웨이트를 종료
            EC.presence_of_element_located( (By.CLASS_NAME,'oTravelBox'))
            )
except Exception as e:
    print('오류 발생',e)
# 네이버 영화 목록 클릭 코드
"""
try:
    element = WebDriverWait(driver,10).until(
            # 지정한 한개 요소가 올라오면 웨이트를 종료
            EC.presence_of_element_located( (By.CLASS_NAME,'movie_run section'))
            )
except Exception as e:
    print('오류 발생',e)
"""    

# 암시적 대기 --> DOM이 다 로드 될때까지 대기 하고 로드되면 바로 진행
# 요소를 찾을 특정 시간 동안 DOM 풀림을 지시 예를 들어 10 초이내 라도 발견되면 진행
#driver.implicitly_wait(10)
# 절대적 대기 --> time.sleep(10) --> 클라우드 페어(디도스 방어 솔류션)

# 더보기 눌러서 --> 게시판 진입
driver.find_element_by_css_selector('.oTravelBox>.boxList>.moreBtnWrap>.moreBtn').click()
#네이버에서 로마를 검색하여 가볼만한 곳의 더보기를 클릭하도록 하는 코드
#driver.find_element_by_class_name('go_site').click()
#배구연맹에서 주요 기록 항목을 클릭하는 코드
#driver.find_element_by_xpath('/stats/41100_triple.asp').click()

# 게시판에서 데이터를 가져올 때
# 데이터가 많으면 세션(혹시 로그인을 해서 접근되는 사이트일 경우) 관리
# 특정 단위별로 로그아원 로그인 계속 시도
# 특정 계시물이 사라질 경우 --> 팝업 발생 --> (없는 ....) --> 팝업 처리에 대한 부분 검토
# 게시판을 스캔시 --> 임계점을 모름!! --> 
# 게시판을 스캔해서 --> 메타 정보를 획득 --> loop를 돌려서 일괄적으로 방문 접근 처리

# "searchModule.SetCategoryList(1, '')" 스크립트 실행
"""
try:
    movieItems=driver.find_elements_by_css_selector('.lst_wrap>.lst_detail_t1>li')
    for lst in movieItems:
        print("제목 : {}".format(lst.find_element_by_css_selector('.info_txt1').text))
except Exception as e:
    print("에러 발생!!!\n{}".format(e))
"""

# 16은 임시값, 게시물을 넘어갔을때 현상을 확인하고자 더 넣음       
for page in range(1,2):
    try:
        # 자바 스크립트 구동하기
        driver.execute_script("searchModule.SetCategoryList(%s, '')" % page)
        # 페이지 넘어갈때 2초간 강제로 쉬게 하는것
        time.sleep(2)
        print("%s 페이지 이동" % page)
        boxItems = driver.find_elements_by_css_selector('.oTravelBox>.boxList>li')
        ##################################
        # 여러 사이트에서 정보를 수집할 경우 공통 정보 정의 단계 필요
        # 상품명, 코멘트, 기간1, 기간2, 가격, 평점, 썸네일, 링크(상품상세정보)
        for li in boxItems:
            # 이미지를 링크값을 사용할것인가?
            #직접 다운로드 해서 우리 서버에 업로드(ftp) 할것인가.
#            print('썸네일',li.find_element_by_css_selector('img').get_attribute('src'))
#            print('링크',li.find_element_by_css_selector('a').get_attribute('onclick'))
#            print('상품명 : {}'.format(li.find_element_by_css_selector('h5.proTit').text))
#            print('코멘트 : {}'.format(li.find_element_by_css_selector('.proSub').text))
#            print('가격 : {}'.format(li.find_element_by_css_selector('.proPrice').text))
            area = ''
            #.info-row 하고나서 한번 띄어주어야 한다.
#            for info in li.find_elements_by_css_selector('.info-row .proInfo'):
#                print( info.text )
#            print('='*100)
            # 데이터 모음
            obj = TourInfo(
                    li.find_element_by_css_selector('h5.proTit').text,
                    li.find_element_by_css_selector('.proPrice').text,
                    li.find_elements_by_css_selector('.info-row .proInfo')[1].text,
                    li.find_element_by_css_selector('a').get_attribute('onclick'),
                    li.find_element_by_css_selector('img').get_attribute('src'),
                    )
            tour_list.append( obj )
    except Exception as e1:
        print('오류',e1)

print(tour_list,len(tour_list))

# 수집한 정보 개수를 루프 --> 페이지 방문 --> 콘텐츠 획득(상품상세정보) --> DB