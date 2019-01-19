import pandas as pd
import numpy as np
import matplotlib as mpl
from sklearn.model_selection import train_test_split

# 한글 폰트 안 깨지게하기위한 import
import matplotlib.font_manager as fm

# 가져올 폰트 지정
font_location='E:/글꼴/H2GTRE.TTF'
# 폰트 이름 지정 
font_name=fm.FontProperties(fname=font_location).get_name()
mpl.rc('font',family=font_name)

class LogisticRegression:
    def __init__(self, lr=0.001, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        
        
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        temp = 1+np.exp(-z)
        return 1 / temp
    def __loss(self, h, y):
        return (-y * np.log(h) + (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if(i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
#                print(f'loss: {self.__loss(h, y)} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return (self.predict_prob(X) >= threshold)*1
    
    def predict_score(self,X,Y):
        pred = Y-self.predict(X,0.5)*1
        score = 0
        for loop in range(len(pred)):
            if(pred[loop]!=0):
                score+=1
        return 1-(score/len(pred))
        
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
        Train_X,Test_X,Train_Y,Test_Y=train_test_split(X,Y,test_size=0.2)
        
        model = LogisticRegression()
        model.__init__()
        model.fit(Train_X,Train_Y)
        
        if(gender%2==0):
            print("====================================Results of men's professional team playoffs(Logistic_regression)====================================")
        else:
            print("====================================Results of women's professional team playoffs(Logistic_regression)====================================")
            
        print(Test_Y)
        print(model.predict(Test_X,0.5)*1)
        print("훈련 세트 정확도: {:.3f}".format(model.predict_score(Train_X,Train_Y)))
        print("테스트 세트 정확도: {:.3f}".format(model.predict_score(Test_X,Test_Y)))
        
        order = np.argsort(-abs(model.theta[1:]))
        weight = model.theta[1:]
        
        features_name = features[order]
        variable_weight = weight[order]
        
        print("=====================================Variable weight(Logistis_regression)=====================================")
        for loop in range(len(features)):
            print("{} : {}".format(features_name[loop],variable_weight[loop]))
        