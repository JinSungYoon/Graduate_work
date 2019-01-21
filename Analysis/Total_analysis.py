import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pylab as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# 한글 폰트 안 깨지기 위한 import
import matplotlib.font_manager as fm

# 가져올 폰트 지정
font_location = 'E:/글꼴/H2GTRE.TTF'
# 폰트 이름 지정
font_name = fm.FontProperties(fname=font_location).get_name()
mpl.rc('font',family=font_name)

def show_graph(x,y,tech):    
    
    plt.figure(figsize=(6,5))
    plt.bar(x,abs(y),alpha = 0.5,align='center',
            label = '가중치')
#    plt.step(range(0,len(y)),y,where='mid',
#              label='cumulative explained variance')
    
    plt.xticks(rotation=90)
    plt.ylabel('가중치')
    plt.xlabel('경기요인')
    plt.title('%s로 분석한 경기요인 가중치'%(tech))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
#    plt.bar(np.arange(len(x)),height,bar_width,
#                  alpha = opacity,
#                  color = '#000FF4',
#                  label = "플레이오프 진출요인 가중치")
#    plt.xlabel('경기요인')
#    plt.ylabel('가중치')
#    plt.title("%s로 분석한 플레이오프 진출 경기요인 가중치"%(skill))
#    
#    plt.xticks(y,x,rotation=90)
#    plt.legend()
#    
#    plt.tight_layout()
#    plt.show()

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
        
        # 자동으로 트레이닝 셋과 테스트셋 나눈 코드
        X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
        
        features = X.columns
        
        # Logistic_regression으로 데이터 훈련
        Lmodel = LogisticRegression()
        Lmodel = Lmodel.fit(X_train,y_train)
        
        # SVM으로 데이터 훈련
        Smodel = SVC(kernel="linear")
        Smodel.fit(X_train,y_train)
        
        # Decision tree로 데이터 훈련
        Dmodel = tree.DecisionTreeClassifier(max_depth=10)
        Dmodel = Dmodel.fit(X_train,y_train)
        
        # Randomforest 데이터 훈련작업
        Rmodel = RandomForestClassifier(bootstrap=True,n_estimators=100,max_features=int(np.sqrt(len(Data.columns))),random_state=0)
        Rmodel = Rmodel.fit(X_train,y_train)
        
        # GradientBoosting 데이터 훈련작업
        Gmodel = GradientBoostingClassifier()
        Gmodel = Gmodel.fit(X_train,y_train)
        
        if(gender%2==0):
            print("******************************Results of men's professional team playoffs******************************")
        else:
            print("******************************Results of women's professional team playoffs******************************")
        
        print("****************************** 로지스틱 리그레션 예측 정확도 ******************************")
        print(y_test.values)
        print(Lmodel.predict(X_test)*1)
        print("훈련 세트 정확도: {:.3f}".format(Lmodel.score(X_train,y_train)))
        print("테스트 세트 정확도: {:.3f}".format(Lmodel.score(X_test,y_test)))
        
        print("****************************** 서포트 벡터 머신 예측 정확도 ******************************")
        print(y_test.values)
        print(Smodel.predict(X_test)*1)
        print("훈련 세트 정확도: {:.3f}".format(Smodel.score(X_train,y_train)))
        print("테스트 세트 정확도: {:.3f}".format(Smodel.score(X_test,y_test)))
        
        print("****************************** 의사결정나무 예측 정확도 ******************************")
        print(y_test.values)
        print(Dmodel.predict(X_test))
        print("훈련 세트 정확도: {:.3f}".format(Dmodel.score(X_train, y_train)))
        print("테스트 세트 정확도: {:.3f}".format(Dmodel.score(X_test, y_test)))

        
        print("****************************** 랜던 포레스트 예측 정확도 ******************************")
        print(y_test.values)
        print(Rmodel.predict(X_test))
        print("훈련 세트 정확도: {:.3f}".format(Rmodel.score(X_train, y_train)))
        print("테스트 세트 정확도: {:.3f}".format(Rmodel.score(X_test, y_test)))
        
        print("****************************** 그래디언트 부스팅 예측 정확도 ******************************")
        print(y_test.values)
        print(Gmodel.predict(X_test))
        print("훈련 세트 정확도: {:.3f}".format(Gmodel.score(X_train, y_train)))
        print("테스트 세트 정확도: {:.3f}".format(Gmodel.score(X_test, y_test)))
        
        print("*** Variable weight(LR) ***")
        Lorder = np.argsort(-abs(Lmodel.coef_))
        weight = Lmodel.coef_
        weight = weight[0]
        
        Lname = features[Lorder]
        Lweight = weight[Lorder]
        Lname = Lname[0]
        Lweight = np.round(Lweight[0],3)
        show_graph(Lname,Lweight,"Logistic-regression")
        
#        for loop in range(len(features)):
#            print("{} : {}".format(Lname[loop],Lweight[loop]))
        
        print("*** Variable weigt(SVM) ***")
        Sorder = np.argsort(-abs(Smodel.coef_))
        Sname = features[Sorder]
        Sweight = np.round(Smodel.coef_[0][Sorder],3)
        Sweight = Sweight[0]
        Sname = Sname[0]
        show_graph(Sname,Sweight,"Support-vector-machine")
        
#        for loop in range(len(features)):
#            print("{} : {}".format(Sname[loop],Sweight[loop]))        
        
        print("*** Variable weight(DT) ***")
        Dweight = Dmodel.feature_importances_
        Dorder = np.argsort(-abs(Dweight))
        Dname = features[Dorder]
        Dweight = np.round(Dweight[Dorder],3)
        show_graph(Dname,Dweight,"Decision-Tree")
        
#        for i in range(len(features)):
#            print("{} : {}".format(Dname[i],Dweight[i]))
#        
        print("*** Variable weight(RF) ***")
        Rweight = Rmodel.feature_importances_
        Rorder = np.argsort(-abs(Rweight))
        Rname = features[Rorder]
        Rweight = np.round(Rweight[Rorder],3)
        show_graph(Rname,Rweight,"Random-Forest-Classifier")
        
        print("*** Variable weight(GB) ***")
        Gweight = Gmodel.feature_importances_
        Gorder = np.argsort(-abs(Gweight))
        
        Gname = features[Gorder]
        Gweight = np.round(Gweight[Gorder],3)
        show_graph(Gname,Gweight,"Gradient-Boosting-Classifier")