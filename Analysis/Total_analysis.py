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
    rank_score = np.zeros(len(origin))
    for index in range(len(origin)):
        for loop in range(len(order)):
            if(origin[index]==order[loop]):
                rank_score[index]+=len(origin)-loop+1
                break
    return rank_score

Mrank = np.zeros(19)
Frank = np.zeros(19)
Macc = [[]*1 for i in range(10)]
Facc = [[]*1 for i in range(10)]


for loop in range(5):
    for gender in range(2):
        if gender%2==0:
            Data = pd.read_pickle("E:/대학교/졸업/졸업작품/분석연습/Extract_M_Data")
#            Data = pd.read_pickle("Male_data")
        else:
            Data = pd.read_pickle("E:/대학교/졸업/졸업작품/분석연습/Extract_F_Data")
#            Data = pd.read_pickle("Female_data")
                    
        # 플레이오프 진출 여부 columns와 결정요소들을 분리한다.
        Y = Data["플레이오프진출"]
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
        Nmodel = MLPClassifier(hidden_layer_sizes=(1,),activation='logistic',solver='lbfgs')
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
        
        if(gender%2==0):
            print("******************************Results of men's professional team playoffs******************************")
        else:
            print("******************************Results of women's professional team playoffs******************************")
        
        print("****************************** 의사결정나무 예측 정확도 ******************************")
        print(y_test.values)
        print(Dmodel.predict(X_test))
        print("훈련 세트 정확도: {:.3f}".format(Dmodel.score(X_train, y_train)))
        print("테스트 세트 정확도: {:.3f}".format(Dmodel.score(X_test, y_test)))
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
        
        print("****************************** 로지스틱 리그레션 예측 정확도 ******************************")
        print(y_test.values)
        print(Lmodel.predict(X_test)*1)
        print("훈련 세트 정확도: {:.3f}".format(Lmodel.score(X_train,y_train)))
        print("테스트 세트 정확도: {:.3f}".format(Lmodel.score(X_test,y_test)))
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
        
        print("****************************** 뉴런 네트워크 예측 정확도 ******************************")
        print(y_test.values)
        print(Nmodel.predict(X_test))
        print("훈련 세트 정확도 : {:.3f}".format(Nmodel.score(X_train,y_train)))
        print("테스트 세트 정확도: {:.3f}".format(Nmodel.score(X_test,y_test)))
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
                
        print("****************************** 랜덤 포레스트 예측 정확도 ******************************")
        print(y_test.values)
        print(Rmodel.predict(X_test))
        print("훈련 세트 정확도: {:.3f}".format(Rmodel.score(X_train, y_train)))
        print("테스트 세트 정확도: {:.3f}".format(Rmodel.score(X_test, y_test)))
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
        
        print("****************************** 서포트 벡터 머신 예측 정확도 ******************************")
        print(y_test.values)
        print(Smodel.predict(X_test)*1)
        print("훈련 세트 정확도: {:.3f}".format(Smodel.score(X_train,y_train)))
        print("테스트 세트 정확도: {:.3f}".format(Smodel.score(X_test,y_test)))
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
        print("*** Variable weight(DT) ***")
        Dweight = Dmodel.feature_importances_
        Dorder = np.argsort(-abs(Dweight))
        Dname = features[Dorder]
        Dweight = np.round(Dweight[Dorder],3)
        if(gender%2==0):
            Mrank+=cal_rank(features,Dname)
        else:
            Frank+=cal_rank(features,Dname)
        show_graph(Dname,Dweight,"Decision-Tree",gender)
        
#        for i in range(len(features)):
#            print("{} : {}".format(Dname[i],Dweight[i]))
#
        
        print("*** Variable weight(LR) ***")
        Lorder = np.argsort(-abs(Lmodel.coef_))
        weight = Lmodel.coef_
        weight = weight[0]
        
        Lname = features[Lorder]
        Lweight = weight[Lorder]
        Lname = Lname[0]
        Lweight = np.round(Lweight[0],3)
        if(gender%2==0):
            Mrank+=cal_rank(features,Lname)
        else:
            Frank+=cal_rank(features,Lname)
        show_graph(Lname,Lweight,"Logistic-regression",gender)
        
        print("*** Variable weight(NN) ***")
        Nweight = Nmodel.coefs_[0].flatten()
        Norder = np.argsort(-abs(Nweight))
        Nname = features[Norder]
        Nweight = np.round(Nweight[Norder],3)
        
        if(gender%2==0):
            Mrank+=cal_rank(features,Nname)
        else:
            Frank+=cal_rank(features,Nname)
        show_graph(Nname,Nweight,"Neural-network-Classifier",gender)
                           
        print("*** Variable weight(RF) ***")
        Rweight = Rmodel.feature_importances_
        Rorder = np.argsort(-abs(Rweight))
        Rname = features[Rorder]
        Rweight = np.round(Rweight[Rorder],3)
        if(gender%2==0):
            Mrank+=cal_rank(features,Rname)
        else:
            Frank+=cal_rank(features,Rname)
        show_graph(Rname,Rweight,"Random-Forest-Classifier",gender)
        
        print("*** Variable weigt(SVM) ***")
        Sorder = np.argsort(-abs(Smodel.coef_))
        Sname = features[Sorder]
        Sweight = np.round(Smodel.coef_[0][Sorder],3)
        Sweight = Sweight[0]
        Sname = Sname[0]
        if(gender%2==0):
            Mrank+=cal_rank(features,Sname)
        else:
            Frank+=cal_rank(features,Sname)
        show_graph(Sname,Sweight,"Support-vector-machine",gender)
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
        
        if(gender%2==0):
            order = np.argsort(Mrank)
            print(features[order])
            print(Mrank[order])
        else:
            order = np.argsort(Frank)
            print(features[order])
            print(Frank[order])
        

print("*******************최종 플레이오프 진출 기여도가 높은 순서*******************")
Morder = np.argsort(-Mrank)
print(features[Morder])
print(Mrank[Morder])
show_graph(features[Morder],Mrank[Morder],"5가지 모델",0)

Forder = np.argsort(-Frank)
print(features[Forder])
print(Frank[Forder])
show_graph(features[Forder],Frank[Forder],"5가지 모델",1)

print("각 모델에 대한 평균 예측률")
#print(Macc)
#print(Facc)
print(np.mean(Macc,axis=1))
print(np.mean(Facc,axis=1))
# 참고한 사이트
# Logistic_regression
#       https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# SVM
#       https://bskyvision.com/163
#       https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
