import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

# 한글 폰트 안 깨지게하기위한 import
import matplotlib.font_manager as fm

# 가져올 폰트 지정
font_location='E:/글꼴/H2GTRE.TTF'
# 폰트 이름 지정 
font_name=fm.FontProperties(fname=font_location).get_name()
mpl.rc('font',family=font_name)

# 플레이오프 진출 요인들 담을 리스트
M_factor_list = [[] for i in range(5)]
F_factor_list = [[] for i in range(5)]

for year in range(13,18):
    kovo_Mresult_table = pd.read_pickle('Kovo_Male_result_table(%s-%s)'%(str(year),str(year+1)))
    kovo_Fresult_table = pd.read_pickle('Kovo_Female_result_table(%s-%s)'%(str(year),str(year+1)))
    
    #kovo_Mresult_table_17 = pd.read_pickle('Kovo_Male_result_table(17-18)')
    #kovo_Fresult_table_17 = pd.read_pickle('Kovo_Female_result_table(17-18)')
    #kovo_Mresult_table_16 = pd.read_pickle('Kovo_Male_result_table(16-17)')
    #kovo_Fresult_table_16 = pd.read_pickle('Kovo_Female_result_table(16-17)')
    #kovo_Mresult_table_15 = pd.read_pickle('Kovo_Male_result_table(15-16)')
    #kovo_Fresult_table_15 = pd.read_pickle('Kovo_Female_result_table(15-16)')
    #kovo_Mresult_table_14 = pd.read_pickle('Kovo_Male_result_table(14-15)')
    #kovo_Fresult_table_14 = pd.read_pickle('Kovo_Female_result_table(14-15)')
    #kovo_Mresult_table_13 = pd.read_pickle('Kovo_Male_result_table(13-14)')
    #kovo_Fresult_table_13 = pd.read_pickle('Kovo_Female_result_table(13-14)')
    
    import Analysis_practice as As
    
    # 임시로 플레이오프 진출한 팀에 대한 내용을 추가했다.
    Male_play_off = []
    Female_play_off = []
    
    #print(len(kovo_Mresult_table))
    for index in range(len(kovo_Mresult_table)) :
        if index<3:
            Male_play_off.append(1)
        else:
            if index==3 and kovo_Mresult_table.iloc[2]["승점"]-kovo_Mresult_table.iloc[3]["승점"]<=3:
                Male_play_off.append(1)
            else:
                Male_play_off.append(0)
    
    for index in range(len(kovo_Fresult_table)) :
        if index<3:
            Female_play_off.append(1)
        else:
            Female_play_off.append(0)
    
    kovo_Mresult_table["플레이오프진출"] = Male_play_off
    kovo_Fresult_table["플레이오프진출"] = Female_play_off
    
    #kovo_Mresult_table_17["플레이오프진출"] = [1,1,1,0,0,0,0]
    #kovo_Fresult_table_17["플레이오프진출"] = [1,1,1,0,0,0]
    #kovo_Mresult_table_16["플레이오프진출"] = [1,1,1,0,0,0,0]
    #kovo_Fresult_table_16["플레이오프진출"] = [1,1,1,0,0,0]
    #kovo_Mresult_table_15["플레이오프진출"] = [1,1,1,0,0,0,0]
    #kovo_Fresult_table_15["플레이오프진출"] = [1,1,1,0,0,0]
    #kovo_Mresult_table_14["플레이오프진출"] = [1,1,1,0,0,0,0]
    #kovo_Fresult_table_14["플레이오프진출"] = [1,1,1,0,0,0]
    #kovo_Mresult_table_13["플레이오프진출"] = [1,1,1,0,0,0,0]
    #kovo_Fresult_table_13["플레이오프진출"] = [1,1,1,0,0,0]
    
    # 데이터프레임의 columns인덱스를 데이터프레임이름.ix[(행요소),(열요소)]하면 인덱스 번호로 접근 가능.
    #print(kovo_result_table.ix[:,7])
    
    # 플레이오프 진출과 다른 요인들이 어느떠한 관계가 있는지 확인
    
    for index in range(131):
        if type(kovo_Mresult_table.ix[0,index])==str:
            continue
        else:
            Co_point = As.correlation(kovo_Mresult_table["플레이오프진출"],kovo_Mresult_table.ix[:,index])
            if abs(Co_point)>0.75:
                M_factor_list[year-13].append((kovo_Mresult_table.columns[index],Co_point))
    #            print("{} Correlation : {}".format(kovo_Mresult_table.columns[index],Co_point))
    
    for index in range(131):
        if type(kovo_Fresult_table.ix[0,index])==str:
            continue
        else:
            Co_point = As.correlation(kovo_Fresult_table["플레이오프진출"],kovo_Fresult_table.ix[:,index])
            if abs(Co_point)>0.75:
                F_factor_list[year-13].append((kovo_Fresult_table.columns[index],Co_point))
    #            print("{} Correlation : {}".format(kovo_Fresult_table.columns[index],Co_point))
            
    
                            
    #plt.plot(kovo_result_table["플레이오프진출"],kovo_result_table[('득점','득점')],'r+',alpha=0.5)
    #plt.axis([0,max(kovo_result_table[('공격','범실')])+10,0,max(kovo_result_table[('공격','순위')])+10])
    #plt.show()

print(M_factor_list)
print(F_factor_list)