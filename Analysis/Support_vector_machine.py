import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

for test in range(5):
    for gender in range(2):
        if gender%2==0:
            Data = pd.read_pickle("E:/대학교/졸업/졸업작품/분석연습/Extract_M_Data")
            #Data = pd.read_pickle("Male_data")
        else:
            Data = pd.read_pickle("E:/대학교/졸업/졸업작품/분석연습/Extract_F_Data")
            #Data = pd.read_pickle("Female_data")
        
        # 플레이오프 진출 여부 columns와 결정요소들을 분리한다.
        Y = Data["플레이오프진출"]
        X = Data.drop("플레이오프진출",axis=1)
        
        features = X.columns
        
        X = X.values
        Y = Y.values
                
        # 자동으로 트레이닝 셋과 테스트셋 나눈 코드
        X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
        
        Smodel = SVC(kernel="linear")
        Smodel.fit(X_train,y_train)
        
        if(gender%2==0):
            print("====================================Results of men's professional team playoffs(SVM)====================================")
        else:
            print("===================================Results of women's professional team playoffs(SVM)====================================")
        print(y_test)
        print(Smodel.predict(X_test)*1)
        print("훈련 세트 정확도: {:.3f}".format(Smodel.score(X_train,y_train)))
        print("테스트 세트 정확도: {:.3f}".format(Smodel.score(X_test,y_test)))
        
        print("====================================Variable weigt(SVM)====================================")
        order = np.argsort(-abs(Smodel.coef_))
        Order_of_variable_name = features[order]
        Order_of_variable_weight = Smodel.coef_[0][order]
        
        for loop in range(len(Order_of_variable_name[0])):
            print("{} : {}".format(Order_of_variable_name[0][loop],Order_of_variable_weight[0][loop]))        


##Load the data
#Extract = pd.read_pickle("E:/대학교/졸업/졸업작품/분석연습/Extract_M_Data")
#
## Data Preprocessing
#X = Extract.drop("플레이오프진출",axis=1)
#y = Extract["플레이오프진출"]
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
#
## Training the Algorithm
#model = SVC(kernel="linear")
#model.fit(X_train,y_train)
#y_pred = model.predict(X_test)
#
## Evaluation the Algoritm
#print("배구 플레이오프 진출에 대한 SVM 결과")
#print(model.coef_)
#print(classification_report(y_test,y_pred))

## Load data
#bankdata = pd.read_csv("E:/대학교/졸업/졸업작품/분석연습/bankdata.csv",engine="python")
#
## Data Preprocessing
#X = bankdata.drop('Class',axis=1)
#y = bankdata['Class']
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
#
## Training the Algorithm
#model = SVC(kernel="linear")
#model.fit(X_train,y_train)
#y_pred = model.predict(X_test)
#
## Evaluation the Algorithm
#print("은행 클래스에 대한 SVM 결과")
#print(classification_report(y_test,y_pred))