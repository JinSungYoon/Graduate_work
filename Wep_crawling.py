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
from selenium import webdriver as wd

from selenium.webdriver.common.by import By
# 명시적 대기를 위해서
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 사전에 필요한 정보를 로드 --> 디비 혹은 쉘, 배치 파일에서 인자로 받아서 세팅
keyword = '로마'
enter='영화'
main_url = 'http://air.interpark.com/'
test_url='http://www.naver.com/'
kovo_url = 'http://www.kovo.co.kr/'

# 드라이버 로드
driver = wd.Chrome(executable_path='Chromedriver.exe')
# 차후 --> 옵셕 부여하여(프록시, 에이전트 조작, 이미지를 배제 )

# 크롤링을 오래 돌리면 --> 임시파일들이 쌓인다!! --> Temp파일 삭제

# 사이트 접속 (get)
#driver.get(main_url)
#driver.get(test_url)
driver.get(kovo_url)

# 검색창을 찾아서 검색어를 입력
# id : SearchGNBText
#driver.find_element_by_id('SearchGNBText').send_keys(keyword)
# 네이버 검색창 id : query
#driver.find_element_by_id('query').send_keys(enter)

# 수정할 경우 --> 뒤에 내용이 붙어버림 --> .clear() --> .send_keys('내용')

# 검색버튼을 클릭
# 버튼의 id가 없으므로 버튼의 클래스 명을 넣는다.
#driver.find_element_by_css_selector('button.search-btn').click()
# 네이버는 id가 있으므로 id로 접근
#driver.find_element_by_id('search_btn').click()
# 배구영맹 STATS를 클릭해야 한다
driver.find_element_by_class_name('t_nav4').click()


# 잠시 대기 --> 페이지가 로드되고 나서 즉각적으로 데이터를 획득하는 행외는 자제->
# 명시적 대기 --> 특정 요소가 로케이트(발견될때까지) 대기
#try:
#    element = WebDriverWait(driver,10).until(
#            # 지정한 한개 요소가 올라오면 웨이트를 종료
#            EC.presence_of_element_located( (By.CLASS_NAME,'oTravelBox'))
#            )
#except Exception as e:
#    print('오류 발생',e)
#try:
#    element = WebDriverWait(driver,10).until(
#            # 지정한 한개 요소가 올라오면 웨이트를 종료
#            EC.presence_of_element_located( (By.CLASS_NAME,'movie_run section'))
#            )
#except Exception as e:
#    print('오류 발생',e)

# 암시적 대기 --> DOM이 다 로드 될때까지 대기 하고 로드되면 바로 진행
# 요소를 찾을 특정 시간 동안 DOM 풀림을 지시 예를 들어 10 초이내 라도 발견되면 진행
driver.implicitly_wait(10)
# 절대적 대기 --> time.sleep(10) --> 클라우드 페어(디도스 방어 솔류션)

# 더보기 눌러서 --> 게시판 진입
#driver.find_element_by_css_selector('.oTravelBox>.boxList>.moreBtnWrap>.moreBtn').click()
#네이버에서 로마를 검색하여 가볼만한 곳의 더보기를 클릭하도록 하는 코드
#driver.find_element_by_class_name('go_site').click()
#배구연맹에서 주요 기록 항목을 클릭하는 코드
#driver.find_element_by_link_text('/stats/41102_triple.asp').click()
driver.find_element_by_xpath('/stats/41100_triple.asp').click()