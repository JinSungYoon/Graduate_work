import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pylab as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# 파이썬 구조체 선언하기
from collections import namedtuple

# 한글 폰트 안 깨지기 위한 import
import matplotlib.font_manager as fm

# 가져올 폰트 지정
font_location = 'E:/글꼴/H2GTRE.TTF'
# 폰트 이름 지정
font_name = fm.FontProperties(fname=font_location).get_name()
mpl.rc('font',family=font_name)

def show_graph(x,y,tech,g):    
    
    plt.figure(figsize=(6,5))
    if(g==0):
        plt.bar(x,abs(y),alpha = 0.5,align='center',
            label = '가중치',color='b')
    else:
        plt.bar(x,abs(y),alpha = 0.5,align='center',
            label = '가중치',color='r')
    
#    plt.step(range(0,len(y)),y,where='mid',
#              label='cumulative explained variance')
    
    plt.xticks(rotation=90)
    plt.ylabel('가중치')
    plt.xlabel('경기요인')
    plt.title('%s로 분석한 경기요인 가중치'%(tech))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
def cal_rank(origin,order):
    rank_score = [0 for i in range(len(origin))]
    for index in range(len(origin)):
        for loop in range(len(order)):
            if(origin[index]==order[loop]):
                rank_score[index]+=len(origin)-loop+1
                break
    return rank_score

#Mrank = np.zeros(27)
#Frank = np.zeros(27)

Mranklog = []
Franklog = []

Macc = [[]*1 for i in range(10)]
Facc = [[]*1 for i in range(10)]

def Analysis_Data(Mdata,Fdata,s,e):
    for loop in range(5):
        for gender in range(s,e):
            if gender%2==0:
                Data = Mdata
                Mrank = np.zeros(len(Mdata.columns)-1)
            else:
                Data = Fdata
                Frank = np.zeros(len(Fdata.columns)-1)
                        
            # 플레이오프 진출 여부 columns와 결정요소들을 분리한다.
            Y = Data.loc[:,"플레이오프진출"]
            X = Data.drop("플레이오프진출",axis=1)
                        
            train = []
            test = []
            num = 0
            
            for i in range(len(Data)-1):
                if(i%5==loop):
                    test.append(i)
                else:
                    train.append(i)
            X_train = X.iloc[train]
            y_train = Y.iloc[train]
            X_test = X.iloc[test]
            y_test = Y.iloc[test]
            # 자동으로 트레이닝 셋과 테스트셋 나눈 코드
    #        X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
    #        print(X_test)
    #        print(X_testp)
            features = X.columns
    #        rank = np.zeros(len(features))
            
            # Decision tree로 데이터 훈련
            max_dep = int(np.sqrt(len(X_train.columns)))
            Dmodel = tree.DecisionTreeClassifier(max_depth=max_dep)
            Dmodel = Dmodel.fit(X_train,y_train)
            
            # Logistic_regression으로 데이터 훈련
            # For small datasets,'liblinear' is a good choice
            # If the  option chosen is 'ovr' then a binary problem is fit for each label.
            # for liblinear and lbfgs solvers set verbose to any positive number for verbosity.
            Lmodel = LogisticRegression(solver="liblinear",multi_class="ovr",verbose=1)
            Lmodel = Lmodel.fit(X_train,y_train)
            
            # Neural Network 데이터 훈련
            Nmodel = MLPClassifier(hidden_layer_sizes=(1,),activation='logistic',solver='lbfgs',alpha=0.0001)
            Nmodel = Nmodel.fit(X_train,y_train)
            
            # Randomforest 데이터 훈련작업
            Rmodel = RandomForestClassifier(bootstrap=True,n_estimators=100,max_features=max_dep,random_state=0)
            Rmodel = Rmodel.fit(X_train,y_train)
            
            # SVM으로 데이터 훈련
            # Kernel coefficient for 'rbf','poly' and 'sigmoid'
            # the original one-vs-one decision function
            # To use a coef_ function you must use linear kernel.
            Smodel = SVC(C=1.0,kernel="linear",gamma="auto",decision_function_shape="ovo")
            Smodel.fit(X_train,y_train)
            
            """
            # GradientBoosting 데이터 훈련작업
            Gmodel = GradientBoostingClassifier()
            Gmodel = Gmodel.fit(X_train,y_train)
            """      
            
#            if(gender%2==0):
#                print("******************************Results of men's professional team playoffs******************************")
#            else:
#                print("******************************Results of women's professional team playoffs******************************")
            
#            print("****************************** 의사결정나무 예측 정확도 ******************************")
#            print(y_test.values)
#            print(Dmodel.predict(X_test))
#            print("훈련 세트 정확도: {:.3f}".format(Dmodel.score(X_train, y_train)))
#            print("테스트 세트 정확도: {:.3f}".format(Dmodel.score(X_test, y_test)))
            if(gender%2==0):
                Macc[num].append(Dmodel.score(X_train,y_train))
                num+=1
                Macc[num].append(Dmodel.score(X_test,y_test))
                num+=1
            else:
                Facc[num].append(Dmodel.score(X_train,y_train))
                num+=1
                Facc[num].append(Dmodel.score(X_test,y_test))
                num+=1
            
#            print("****************************** 로지스틱 리그레션 예측 정확도 ******************************")
#            print(y_test.values)
#            print(Lmodel.predict(X_test)*1)
#            print("훈련 세트 정확도: {:.3f}".format(Lmodel.score(X_train,y_train)))
#            print("테스트 세트 정확도: {:.3f}".format(Lmodel.score(X_test,y_test)))
            if(gender%2==0):
                Macc[num].append(Lmodel.score(X_train,y_train))
                num+=1
                Macc[num].append(Lmodel.score(X_test,y_test))
                num+=1
            else:
                Facc[num].append(Lmodel.score(X_train,y_train))
                num+=1
                Facc[num].append(Lmodel.score(X_test,y_test))
                num+=1
            
#            print("****************************** 뉴런 네트워크 예측 정확도 ******************************")
#            print(y_test.values)
#            print(Nmodel.predict(X_test))
#            print("훈련 세트 정확도 : {:.3f}".format(Nmodel.score(X_train,y_train)))
#            print("테스트 세트 정확도: {:.3f}".format(Nmodel.score(X_test,y_test)))
            if(gender%2==0):
                Macc[num].append(Nmodel.score(X_train,y_train))
                num+=1
                Macc[num].append(Nmodel.score(X_test,y_test))
                num+=1
            else:
                Facc[num].append(Nmodel.score(X_train,y_train))
                num+=1
                Facc[num].append(Nmodel.score(X_test,y_test))
                num+=1
                    
#            print("****************************** 랜덤 포레스트 예측 정확도 ******************************")
#            print(y_test.values)
#            print(Rmodel.predict(X_test))
#            print("훈련 세트 정확도: {:.3f}".format(Rmodel.score(X_train, y_train)))
#            print("테스트 세트 정확도: {:.3f}".format(Rmodel.score(X_test, y_test)))
            if(gender%2==0):
                Macc[num].append(Rmodel.score(X_train,y_train))
                num+=1
                Macc[num].append(Rmodel.score(X_test,y_test))
                num+=1
            else:
                Facc[num].append(Rmodel.score(X_train,y_train))
                num+=1
                Facc[num].append(Rmodel.score(X_test,y_test))
                num+=1
            
#            print("****************************** 서포트 벡터 머신 예측 정확도 ******************************")
#            print(y_test.values)
#            print(Smodel.predict(X_test)*1)
#            print("훈련 세트 정확도: {:.3f}".format(Smodel.score(X_train,y_train)))
#            print("테스트 세트 정확도: {:.3f}".format(Smodel.score(X_test,y_test)))
            if(gender%2==0):
                Macc[num].append(Smodel.score(X_train,y_train))
                num+=1
                Macc[num].append(Smodel.score(X_test,y_test))
                num+=1
            else:
                Facc[num].append(Smodel.score(X_train,y_train))
                num+=1
                Facc[num].append(Smodel.score(X_test,y_test))
                num+=1
            """
            print("****************************** 그래디언트 부스팅 예측 정확도 ******************************")
            print(y_test.values)
            print(Gmodel.predict(X_test))
            print("훈련 세트 정확도: {:.3f}".format(Gmodel.score(X_train, y_train)))
            print("테스트 세트 정확도: {:.3f}".format(Gmodel.score(X_test, y_test)))
            if(gender%2==0):
                Macc[num].append(Gmodel.score(X_train,y_train))
                num+=1
                Macc[num].append(Gmodel.score(X_test,y_test))
                num+=1
            else:
                Facc[num].append(Gmodel.score(X_train,y_train))
                num+=1
                Facc[num].append(Gmodel.score(X_test,y_test))
                num+=1
            """
#            print("*** Variable weight(DT) ***")
            Dweight = Dmodel.feature_importances_
            Dorder = np.argsort(-abs(Dweight))
            Dname = features[Dorder]
            Dweight = np.round(Dweight[Dorder],3)
            if(gender%2==0):
                Mrank+=cal_rank(features,Dname)
                Mranklog.append(cal_rank(features,Dname))
            else:
                Frank+=cal_rank(features,Dname)
                Franklog.append(cal_rank(features,Dname))
#            show_graph(Dname,Dweight,"Decision-Tree",gender)
            
#            print("*** Variable weight(LR) ***")
            Lorder = np.argsort(-abs(Lmodel.coef_))
            weight = Lmodel.coef_
            weight = weight[0]
            
            Lname = features[Lorder]
            Lweight = weight[Lorder]
            Lname = Lname[0]
            Lweight = np.round(Lweight[0],3)
            if(gender%2==0):
                Mrank+=cal_rank(features,Lname)
                Mranklog.append(cal_rank(features,Lname))
            else:
                Frank+=cal_rank(features,Lname)
                Franklog.append(cal_rank(features,Lname))
#            show_graph(Lname,Lweight,"Logistic-regression",gender)
            
#            print("*** Variable weight(NN) ***")
            Nweight = Nmodel.coefs_[0].flatten()
            Norder = np.argsort(-abs(Nweight))
            Nname = features[Norder]
            Nweight = np.round(Nweight[Norder],3)
            
            if(gender%2==0):
                Mrank+=cal_rank(features,Nname)
                Mranklog.append(cal_rank(features,Nname))
            else:
                Frank+=cal_rank(features,Nname)
                Franklog.append(cal_rank(features,Nname))
#            show_graph(Nname,Nweight,"Neural-network-Classifier",gender)
                               
#            print("*** Variable weight(RF) ***")
            Rweight = Rmodel.feature_importances_
            Rorder = np.argsort(-abs(Rweight))
            Rname = features[Rorder]
            Rweight = np.round(Rweight[Rorder],3)
            if(gender%2==0):
                Mrank+=cal_rank(features,Rname)
                Mranklog.append(cal_rank(features,Rname))
            else:
                Frank+=cal_rank(features,Rname)
                Franklog.append(cal_rank(features,Rname))
#            show_graph(Rname,Rweight,"Random-Forest-Classifier",gender)
            
#            print("*** Variable weigt(SVM) ***")
            Sorder = np.argsort(-abs(Smodel.coef_))
            Sname = features[Sorder]
            Sweight = np.round(Smodel.coef_[0][Sorder],3)
            Sweight = Sweight[0]
            Sname = Sname[0]
            if(gender%2==0):
                Mrank+=cal_rank(features,Sname)
                Mranklog.append(cal_rank(features,Sname))
            else:
                Frank+=cal_rank(features,Sname)
                Franklog.append(cal_rank(features,Sname))
#            show_graph(Sname,Sweight,"Support-vector-machine",gender)
            """
            print("*** Variable weight(GB) ***")
            Gweight = Gmodel.feature_importances_
            Gorder = np.argsort(-abs(Gweight))
            Gname = features[Gorder]
            Gweight = np.round(Gweight[Gorder],3)
            if(gender%2==0):
                Mrank+=cal_rank(features,Gname)
            else:
                Frank+=cal_rank(features,Gname)
            show_graph(Gname,Gweight,"Gradient-Boosting-Classifier",gender)
            """
            
#            if(gender%2==0):
#                order = np.argsort(Mrank)
#                print(features[order])
#                print(Mrank[order])
#            else:
#                order = np.argsort(Frank)
#                print(features[order])
#                print(Frank[order])       
    
#    Mfeatures = Mdata.columns
#    Ffeatures = Fdata.columns
    
#    print("*******************최종 플레이오프 진출 기여도가 높은 순서*******************")
#    Morder = np.argsort(-Mrank)
#    print(Morder)
#    print(Mfeatures[Morder])
#    print(Mrank[Morder])
#    print(Mfeatures)
#    print(cal_rank(Mfeatures,Mfeatures[Morder]))
#    show_graph(Mfeatures[Morder],Mrank[Morder],"5가지 모델",0)
    
#    Forder = np.argsort(-Frank)
#    print(Forder)
#    print(Ffeatures[Forder])
#    print(Frank[Forder])
#    print(Ffeatures)
#    print(cal_rank(Ffeatures,Ffeatures[Forder]))
#    show_graph(Ffeatures[Forder],Frank[Forder],"5가지 모델",1)
    
    train_acc = []
    
    print("각 모델에 대한 평균 예측률")
#    print(np.mean(Macc,axis=1))
    print(np.mean(Facc,axis=1))
    
    print("전체 모델에 대한 평균 예측률")
    
    def cal_mean(table):
        train=[]
        test=[]
        for it in range(len(table)):
            if(it%2==0):
                train.append(table[it])
            else:
                test.append(table[it])
        print("훈련모델의 평균정확도:{:0.2f}".format(np.mean(train)))
        print("검정모델의 평균정확도:{:0.2f}".format(np.mean(test)))
        return np.mean(test)
    
#    train_acc.append(cal_mean(Macc))
    train_acc.append(cal_mean(Facc))
    return train_acc

Mdata = pd.read_pickle("E:/대학교/졸업/졸업작품/분석연습/Extract_M_Data")
Fdata = pd.read_pickle("E:/대학교/졸업/졸업작품/분석연습/Extract_F_Data")

var = [#['공격_시도', '공격_성공', '공격_공격차단', '공격_범실','공격_성공률','공격_효율'],
       ['오픈공격_시도', '오픈공격_성공', '오픈공격_공격차단', '오픈공격_범실','오픈공격_성공률','오픈공격_효율'],
       ['시간차공격_시도', '시간차공격_성공', '시간차공격_공격차단', '시간차공격_범실','시간차공격_성공률','시간차공격_효율'],
       ['이동공격_시도', '이동공격_성공', '이동공격_공격차단', '이동공격_범실', '이동공격_성공률'],
       ['후위공격_시도', '후위공격_성공', '후위공격_공격차단', '후위공격_범실', '후위공격_성공률','후위공격_효율'],
       ['속공_시도','속공_성공', '속공_공격차단', '속공_범실', '속공_성공률','속공_효율'],
       ['퀵오픈_시도', '퀵오픈_성공', '퀵오픈_공격차단','퀵오픈_범실', '퀵오픈_성공률','퀵오픈_효율'],
       ['서브_시도', '서브_성공', '서브_범실', '서브_세트당평균','서브_효율'],
       ['블로킹_시도','블로킹_성공', '블로킹_유효블락', '블로킹_실패', '블로킹_범실', '블로킹_어시스트', '블로킹_세트당평균','블로킹_효율'],
       ['디그_시도', '디그_성공', '디그_실패', '디그_범실', '디그_세트당평균','디그_효율'],
       ['세트_시도', '세트_성공','세트_범실', '세트_세트당평균','세트_효율'],
       ['리시브_시도', '리시브_정확', '리시브_범실', '리시브_리시브평균','리시브_효율']]

class Info:
    def __init__(self,var_name,acc):
        self.var_name = var_name
        self.acc = acc
    def __repr__(self):
        return repr((self.var_name,self.acc))

result = []
total = int(input("전체 갯수를 입력하세요 : "))
sub = int(input("선택할 갯수를 입력하세요 : "))
# '공격', '오픈공격','시간차공격','이동공격','후위공격','속공','퀵오픈',
var_name = ['오픈공격','시간차공격','이동공격','후위공격','속공','퀵오픈','서브','블로킹','디그','세트','리시브']
count = 0

box = []
name = []
visited = np.zeros(total)

def combie(box,name,var,var_name,s,total,sub,M,F):
    global count
    global result
    if(len(box)==sub):
        for i in range(sub):
            # sum(list,[]) 2차원 배열을 1차원 배열로 바꾸는 방법
            Ex_Mdata = M[sum(box,[])].copy()
            Ex_Fdata = F[sum(box,[])].copy()
            Ex_Mdata.loc[:,"플레이오프진출"] = M["플레이오프진출"]
            Ex_Fdata.loc[:,"플레이오프진출"] = F["플레이오프진출"]
#            print(name)
#            print(Analysis_Data(Ex_Mdata,Ex_Fdata))
#            result.append([name.copy(),Analysis_Data(Ex_Mdata,Ex_Fdata)])
            result.append(Info(name.copy(),Analysis_Data(Ex_Mdata,Ex_Fdata,1,2)))
            count+=1
            return
    else:
        for i in range(s,total):
            if(not visited[i]):
                visited[i] = 1
                box.append(var[i])
                name.append(var_name[i])
                combie(box,name,var,var_name,i,total,sub,M,F)
                visited[i] = 0
                box.pop()
                name.pop()


# .copy()를 사용하지 않으면 원래 테이블에 대해서도 오류가 생겨 SettingWithCopyWarning이 발생
# https://www.youtube.com/watch?v=4R4WsDJ-KVc 참조
#Ex_Mdata = Mdata[var[5]].copy()
#Ex_Fdata = Fdata[var[5]].copy()

#Ex_Mdata.loc[:,"플레이오프진출"] = Mdata["플레이오프진출"]
#Ex_Fdata.loc[:,"플레이오프진출"] = Fdata["플레이오프진출"]

combie(box,name,var,var_name,0,total,sub,Mdata,Fdata)

result.sort(key=lambda x : x.acc,reverse=True)

for loop in range(5):
    print(result[loop])

# 참고한 사이트
# Logistic_regression
#       https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# SVM
#       https://bskyvision.com/163
#       https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
