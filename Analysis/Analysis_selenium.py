import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from collections import Counter
from sklearn import preprocessing

# 한글 폰트 안 깨지게하기위한 import
import matplotlib.font_manager as fm

# 가져올 폰트 지정
font_location='E:/글꼴/H2GTRE.TTF'
# 폰트 이름 지정 
font_name=fm.FontProperties(fname=font_location).get_name()
mpl.rc('font',family=font_name)

# 시즌을 count를 할 갯수
count = 10
# 시작 년도
syear = 8
# =====================================================================7년치 시즌 데이터 ===================================================================

# 플레이오프 진출 요인들 담을 리스트
M_factor_list = [[] for i in range(count)]
F_factor_list = [[] for i in range(count)]
Mcount = []
Fcount = []

Mdata = []
Fdata = []

# 시즌 결과 데이터를 저장한다.
for year in range(syear,18):
    kovo_Mresult_table = pd.read_pickle('Kovo_Male_result_table(%s-%s)'%(str(year),str(year+1)))
    kovo_Fresult_table = pd.read_pickle('Kovo_Female_result_table(%s-%s)'%(str(year),str(year+1)))
    
    # 플레이오프와 관련없는 순위/팀/경기수/세트수에 대한 데이터 제거
    for i in range(7,72):
        if i ==7:   
            if kovo_Mresult_table.columns[i][1]=='순위':
               for index in range(i,i+3):
                   del kovo_Mresult_table[kovo_Mresult_table.columns[i]]
                   del kovo_Fresult_table[kovo_Fresult_table.columns[i]]
        else:
            if kovo_Mresult_table.columns[i][1]=='순위':
               for index in range(i,i+4):
                   del kovo_Mresult_table[kovo_Mresult_table.columns[i]]
                   del kovo_Fresult_table[kovo_Fresult_table.columns[i]]
        
    # 시즌 승패결과를 Season_data를 저장
    Season_male_data = pd.read_pickle('Male_Season(%s-%s)'%(str(year),str(year+1)))
    Season_female_data = pd.read_pickle('Female_Season(%s-%s)'%(str(year),str(year+1)))
    
    # 시즌의 순위를 남녀 Team_name에 저장한다.
    Male_team_name = kovo_Mresult_table.index
    Female_team_name = kovo_Fresult_table.index
    
    # 남년 최다연승 최다연패을 저장할 배열을 생성한다.
    Male_win_score = np.zeros(len(Male_team_name))
    Male_lose_score = np.zeros(len(Male_team_name))
    Female_win_score = np.zeros(len(Female_team_name))
    Female_lose_score = np.zeros(len(Female_team_name))
    win = 0
    lose = 0
    
    # 남자팀의 최다 연승 최다 연패를 계산하여 배열에 저장한다.
    for team in range(len(Male_team_name)):
        for index in range(len(Season_male_data)):
            if Season_male_data["홈"][index] == Male_team_name[team] and Season_male_data["승패"][index] == "승" or Season_male_data["상대팀"][index] == Male_team_name[team] and Season_male_data["승패"][index] == "패":
                win += 1
                lose = 0
                if Male_win_score[team] < win:
                    Male_win_score[team] = win
            elif Season_male_data["홈"][index] == Male_team_name[team] and Season_male_data["승패"][index] == "패" or Season_male_data["상대팀"][index] == Male_team_name[team] and Season_male_data["승패"][index] == "승":
                lose+=1
                win = 0
                if Male_lose_score[team] < lose:
                    Male_lose_score[team] = lose
    
    # 여자팀 최다 연승 최다 연패를 계산하여 저장한다.
    for team in range(len(Female_team_name)):
        for index in range(len(Season_female_data)):
            if Season_female_data["홈"][index] == Female_team_name[team] and Season_female_data["승패"][index] == "승" or Season_female_data["상대팀"][index] == Female_team_name[team] and Season_female_data["승패"][index] == "패":
                win += 1
                lose = 0
                if Female_win_score[team] < win:
                    Female_win_score[team] = win
            elif Season_female_data["홈"][index] == Female_team_name[team] and Season_female_data["승패"][index] == "패" or Season_female_data["상대팀"][index] == Female_team_name[team] and Season_female_data["승패"][index] == "승":
                lose+=1
                win = 0
                if Female_lose_score[team] < lose:
                    Female_lose_score[team] = lose
            

    import Analysis_practice as As
    
    # 임시로 플레이오프 진출한 팀에 대한 내용을 추가했다.
    Male_play_off = []
    Female_play_off = []
    
    #print(len(kovo_Mresult_table))
    for index in range(len(kovo_Mresult_table)) :
        if index<3:
            Male_play_off.append(1)
        else:
            # 11년도 시즌 이후부터는 4등과 3등의 승점이 3점 이내일 경우 플레이오프에 진출하므로 조건을 추가해줘야 한다.
            if year>10 and index==3 and kovo_Mresult_table.iloc[2]["승점"]-kovo_Mresult_table.iloc[3]["승점"]<=3:
                Male_play_off.append(1)
            else:
                Male_play_off.append(0)
    
    for index in range(len(kovo_Fresult_table)) :
        if index<3:
            Female_play_off.append(1)
        else:
            Female_play_off.append(0)
    
    kovo_Mresult_table["최다연승"] = Male_win_score
    kovo_Mresult_table["최다연패"] = Male_lose_score   
    kovo_Fresult_table["최다연승"] = Female_win_score
    kovo_Fresult_table["최다연패"] = Female_lose_score
    kovo_Mresult_table["플레이오프진출"] = Male_play_off
    kovo_Fresult_table["플레이오프진출"] = Female_play_off
    
    if year<=10:
        del kovo_Mresult_table["승률"]
        del kovo_Fresult_table["승률"]
    else:
        del kovo_Mresult_table["승점"]
        del kovo_Fresult_table["승점"]
    
    Mdata.append(kovo_Mresult_table)
    Fdata.append(kovo_Fresult_table)

    # 데이터프레임의 columns인덱스를 데이터프레임이름.ix[(행요소),(열요소)]하면 인덱스 번호로 접근 가능.
    #print(kovo_result_table.ix[:,7])
    
    # 플레이오프 진출과 다른 요인들이 어느떠한 관계가 있는지 확인
    """
    for index in range(len(kovo_Mresult_table.columns)-1):
        # 남자부 각 부문별 상관관계
        if type(kovo_Mresult_table.ix[0,index])==str:
            continue
        else:
            Co_point = As.correlation(kovo_Mresult_table["플레이오프진출"],kovo_Mresult_table.ix[:,index])
            if abs(Co_point)>0.75:
#                M_factor_list[year-syear].append((kovo_Mresult_table.columns[index],Co_point))
                # 각 요소들이 10년치 데이터에 얼마나 있는지 알아보기 위한 과정
#                Mcount.append(kovo_Mresult_table.columns[index])
#                print("{} Correlation : {}".format(kovo_Mresult_table.columns[index],Co_point))
        # 여자부 각 부문별 상관관계 
        if type(kovo_Fresult_table.ix[0,index])==str:
            continue
        else:
            Co_point = As.correlation(kovo_Fresult_table["플레이오프진출"],kovo_Fresult_table.ix[:,index])
            if abs(Co_point)>0.75:
                F_factor_list[year-syear].append((kovo_Fresult_table.columns[index],Co_point))
                # 각 요소들이 몇개씩 있는지 알아보기 위해서 하나의 리스트에 다 넣는과정
#                Fcount.append(kovo_Fresult_table.columns[index])
#                print("{} Correlation : {}".format(kovo_Fresult_table.columns[index],Co_point))     
     
    # 각 부문별 성공률을 플레이오프와 비교해 본다
#    for index in range(len(M_rate)):
#         Co_point = As.correlation(kovo_Fresult_table["플레이오프진출"],M_rate[index][0])
#        if abs(Co_point) > 0.75:
#            M_factor_list[year-syear].append((index,Co_point))
#        Co_point = As.correlation(kovo_Fresult_table["플레이오프진출"],F_rate[index][0])
#        if abs(Co_point) > 0.75:
#            F_factor_list[year-syear].append((index,Co_point))
   
    # 어떠한 항목 실험하는 실험실
    
    Co_point = As.correlation(kovo_Mresult_table["플레이오프진출"],kovo_Mresult_table[("득점","득점")]/kovo_Mresult_table[("득점","세트수")])
    if abs(Co_point) > 0.75:
        M_factor_list[year-13].append(("세트당 득점",Co_point))
    Co_point = As.correlation(kovo_Mresult_table["플레이오프진출"],kovo_Mresult_table[("득점","득점")]/kovo_Mresult_table[("득점","경기수")])
    if abs(Co_point) > 0.75:
        M_factor_list[year-13].append(("경기당 득점",Co_point))
    Co_point = As.correlation(kovo_Fresult_table["플레이오프진출"],kovo_Fresult_table[("득점","득점")]/kovo_Fresult_table[("득점","세트수")])
    if abs(Co_point) > 0.75:
        F_factor_list[year-13].append(("세트당 득점",Co_point))
    Co_point = As.correlation(kovo_Fresult_table["플레이오프진출"],kovo_Fresult_table[("득점","득점")]/kovo_Fresult_table[("득점","경기수")])
    if abs(Co_point) > 0.75:
        F_factor_list[year-13].append(("경기당 득점",Co_point))
    """
    #plt.plot(kovo_result_table["플레이오프진출"],kovo_result_table[('득점','득점')],'r+',alpha=0.5)
    #plt.axis([0,max(kovo_result_table[('공격','범실')])+10,0,max(kovo_result_table[('공격','순위')])+10])
    #plt.show()

#print(M_factor_list)
#print(F_factor_list)

# 각 요소들이 7년치 데이터에서 얼마나 많이 나오는지 알아보기 위함
#print(Counter(Mcount))
#print(Counter(Fcount))

# Male_data와 Female_data로 합친다.
Male_data = pd.concat([Mdata[0],Mdata[1],Mdata[2],Mdata[3],Mdata[4],Mdata[5],Mdata[6],Mdata[7],Mdata[8],Mdata[9]])
Female_data = pd.concat([Fdata[0],Fdata[1],Fdata[2],Fdata[3],Fdata[4],Fdata[5],Fdata[6],Fdata[7],Fdata[8],Fdata[9]])

# 플레이오프와 관계없는 데이터를 제거한다

def delete_feature(name):
    del Male_data[name]
    del Female_data[name]
    
delete_feature('경기수')
delete_feature('순위')
delete_feature('승')
delete_feature('패')
delete_feature('세트득실률')
delete_feature('점수득실률')

Mplayoff = Male_data['플레이오프진출']
del Male_data['플레이오프진출']
Fplayoff = Female_data['플레이오프진출']
del Female_data['플레이오프진출']

def change_name(table):
    for loop in range(0,len(table.columns)):
        # 튜플로 된 이름들은 길이가 2이므로
        if len(table.columns[loop])==2:
            # 득점,벌칙,범실은 2개가 겹치므로 하나만 넣어준다.
            if table.columns[loop][-1]=='득점' or table.columns[loop][-1]=='벌칙':
                Male_data.rename(columns={Male_data.columns[loop]:Male_data.columns[loop][-2]},inplace='True')
                Female_data.rename(columns={Female_data.columns[loop]:Female_data.columns[loop][-2]},inplace='True')
            else:
                Male_data.rename(columns={Male_data.columns[loop]:Male_data.columns[loop][-2]+'_'+Male_data.columns[loop][-1]},inplace='True')
                Female_data.rename(columns={Female_data.columns[loop]:Female_data.columns[loop][-2]+'_'+Female_data.columns[loop][-1]},inplace='True')
   
change_name(Male_data)
change_name(Female_data)

def data_norm(table):
    col = table.columns
    
    # 데이터 정규화 과정
    x = table[col].values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x.astype(float))
    table = pd.DataFrame(x_scaled,
                         columns=col,
                         index=table.index)
    return table

Male_data_norm = data_norm(Male_data)
Female_data_norm = data_norm(Female_data)

# 데이터 가중치 확인하기

def Confirm_feature_weight(table,result):
    # 데이터 전처리 과정
    from sklearn.cross_validation import train_test_split
    
    X,y = table.values,result.values
    
    # 테스트셋을 원래 데이터의 20%만 허용한다.
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    
    # 1. D차원의 데이터간의 연관성을 찾기 위해 데이터를 먼저 표준화 시킨다. (위에서 표준화를 하였으므로 생각한다.)
       
    #sc = StandardScaler()
    #X_train_std = sc.fit_transform(X_train)
    #X_test_std = sc.transform(X_test)

    # 2. 특징들 상호간의 각각의 공분산을 구하기 위해 공분산 행렬을 만든다.
    cov_mat = np.cov(X_train.T) # 공분산 행렬을 생성해주는 함수
    
    # 3. 공분산 행렬을 Eigen value와 Eigen vector로 분해한다.
    # 이것을 Eigendecomposition이라고 한다.
    
    Eval,Evec = np.linalg.eig(cov_mat)
    
    #print("\nEigenvalues : {}".format(M_Eval))
    
    # eigen value의 값이 큰 순서를 E_val_des_order에 저장한다. np.argsort(음수*데이터값)을 넣으면 크기가 큰 숫자부터 1~N까지 나온다.
    E_val_des_order = np.argsort(-Eval)
    
    # 4. 공분산행렬을 통해 그 두가지(Eigen value,Eigen vector)를 유도하는 것이 가능
    tot = sum(Eval)
    
    var_exp = [(i/tot) for i in sorted(Eval,reverse=True)]
    
    # Eigen value / Eigen value의 합을 각각 구한다. 나온 각각의 값은 Eigen value의 설명 분산 비율이다.
    # 즉, 어떤 Eigen value가 가장 설명력이 높은지를 비율로 나타내기 위한 것이다.
    
    cum_var_exp = np.cumsum(var_exp)    # 누적 합을 계산해주는 함수 -> 누적 백분위로 표현
    
#    plt.figure(figsize=(18,8))
#    plt.bar(table.columns[E_val_des_order],var_exp,alpha = 0.5,align='center',
#            label = 'individual explained variance')
#    plt.step(range(0,len(cum_var_exp)),cum_var_exp,where='mid',
#              label='cumulative explained variance')
#    plt.xticks(rotation=90)
#    plt.ylabel('Explained variance ratio')
#    plt.xlabel('Principal components')
#    plt.legend(loc='best')
#    plt.tight_layout()
#    plt.show()
    # 각각의 항목에 대한 weight값을 텍스트로 나타내는것
    weight_order = table.columns[E_val_des_order]
    for loop in range(0,len(table.columns)):
        print("변수:{}\tweight:{}".format(weight_order[loop],cum_var_exp[loop]))

#print("============================남자경기요인============================")
#Confirm_feature_weight(Male_data_norm,Mplayoff)
#print("============================여자경기요인============================")
#Confirm_feature_weight(Female_data_norm,Fplayoff)

"""
# 데이터 전처리 과정
from sklearn.cross_validation import train_test_split

MX,My = Male_data_norm.values,Mplayoff.values

MX_train,MX_test,My_train,My_test = train_test_split(MX,My,test_size=0.2,random_state=0)

# 1. D차원의 데이터간의 연관성을 찾기 위해 데이터를 먼저 표준화 시킨다.

# 2. 특징들 상호간의 각각의 공분산을 구하기 위해 공분산 행렬을 만든다.
cov_mat = np.cov(MX_train.T) # 공분산 행렬을 생성해주는 함수

# 3. 공분산 행렬을 Eigen value와 Eigen vector로 분해한다.
# 이것을 Eigendecomposition이라고 한다.

M_Eval,M_Evec = np.linalg.eig(cov_mat)

#print("\nEigenvalues : {}".format(M_Eval))

# eigen value의 값이 큰 순서를 E_val_des_order에 저장한다. np.argsort(음수*데이터값)을 넣으면 크기가 큰 숫자부터 1~N까지 나온다.
E_val_des_order = np.argsort(-M_Eval)

# 4. 공분산행렬을 통해 그 두가지(Eigen value,Eigen vector)를 유도하는 것이 가능
tot = sum(M_Eval)

var_exp = [(i/tot) for i in sorted(M_Eval,reverse=True)]

# Eigen value / Eigen value의 합을 각각 구한다. 나온 각각의 값은 Eigen value의 설명 분산 비율이다.
# 즉, 어떤 Eigen value가 가장 설명력이 높은지를 비율로 나타내기 위한 것이다.

cum_var_exp = np.cumsum(var_exp)    # 누적 합을 계산해주는 함수 -> 누적 백분위로 표현

plt.figure(figsize=(18,10))
plt.bar(Male_data.columns[E_val_des_order],var_exp,alpha = 0.5,align='center',
        label = 'individual explained variance')
plt.step(range(0,len(cum_var_exp)),cum_var_exp,where='mid',
          label='cumulative explained variance')
plt.xticks(rotation=90)
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
"""
# pickle로 변환한다.
Male_data.to_pickle("Male_data")
Female_data.to_pickle("Female_data")