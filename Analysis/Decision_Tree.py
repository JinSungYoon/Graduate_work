# reference : https://pythonprogramminglanguage.com/decision-tree-visual-example/

import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import pandas as pd
import pandas as pd
import numpy as np

######################################## RandomForest ########################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import export_graphviz

for test in range(5):
    for gender in range(2):
        if gender%2==0:
            Data = pd.read_pickle("Extract_M_Data")
            #Data = pd.read_pickle("Male_data")
        else:
            Data = pd.read_pickle("Extract_F_Data")
            #Data = pd.read_pickle("Female_data")
            
        # 남자 테이블 데이터에서 필요없는 columns를 하나씩 지우는 과정
        
        # 테스트 데이터와 트레이팅 데이터를 담을 리스트를 생성한다.
        Test_X = []
        Test_Y = []
        Train_X = []
        Train_Y = []
        
        # 플레이오프 진출 여부 columns와 결정요소들을 분리한다.
        Y = Data["플레이오프진출"]
        del Data["플레이오프진출"]
        X = Data
        
        # five fold validation에 의해서 트레이닝 세트와 테스트셋 나눈것
        
        # 10배수는 test데이터로 나머지는 training 데이터로 넣는다.
        for loop in range(0,len(Data)):
            if loop%(5+test) == 0:
                Test_Y.append(Y[loop])
                Test_X.append(X.iloc[loop].values)
            else:
                Train_Y.append(Y[loop])
                Train_X.append(X.iloc[loop].values)
        
        # 자동으로 트레이닝 셋과 테스트셋 나눈 코드
#        Train_X,Test_X,Train_Y,Test_Y=train_test_split(X,Y,random_state=1)
        
        data_feature_name = Data.columns
            
        # training data로 데이터를 분류하는 작업
        clf = tree.DecisionTreeClassifier(max_depth=4)
        clf = clf.fit(Train_X,Train_Y)
        
        # Randomforest 데이터 훈련작업
        CLF = RandomForestClassifier(bootstrap=True,n_estimators=100,max_features=int(np.sqrt(len(Data.columns))),random_state=0)
        CLF = CLF.fit(Train_X,Train_Y)
        
#        print("###################################Decision_Tree###################################")
#        print(Test_Y)
#        print(clf.predict(Test_X))
#        print("훈련 세트 정확도: {:.3f}".format(clf.score(Train_X, Train_Y)))
#        print("테스트 세트 정확도: {:.3f}".format(clf.score(Test_X, Test_Y)))
        
        if(gender%2==0):
            print("=====================================Male_data=====================================")
        else:
            print("=====================================Female_data=====================================")
        print(Test_Y)
        print(CLF.predict(Test_X))
        print("훈련 세트 정확도: {:.3f}".format(CLF.score(Train_X, Train_Y)))
        print("테스트 세트 정확도: {:.3f}".format(CLF.score(Test_X, Test_Y)))
        
        importance = CLF.feature_importances_
        order = np.argsort(-importance)
        Order_of_variable_name = Data.columns[order]
        Order_of_variable_weight = importance[order]
        
#        print(Data.columns[order])
#        print(importance[order])
        print("=====================================변수 가중치=====================================")
        for i in range(len(Data.columns)):
            print("{} : {}".format(Order_of_variable_name[i],Order_of_variable_weight[i]))
        
        
        
        # Decision_tree 그래프를 그리는 로직
#        dot_data = tree.export_graphviz(clf,
#                                        feature_names=data_feature_name,
#                                        out_file=None,
#                                        filled=True,
#                                        rounded=True)
#        graph = pydotplus.graph_from_dot_data(dot_data)
#        
#        colors = ('skyblue', 'pink')
#        edges = collections.defaultdict(list)
#        
#        for edge in graph.get_edge_list():
#            edges[edge.get_source()].append(int(edge.get_destination()))
#        
#        for edge in edges:
#            edges[edge].sort()    
#            for i in range(2):
#                dest = graph.get_node(str(edges[edge][i]))[0]
#                dest.set_fillcolor(colors[i])
#        if gender%2==0:
#            graph.write_svg('Random_Male_playoff(%s).svg'%(test))
#        else:
#            graph.write_svg('Random_Female_playoff(%s).svg'%(test))

        from subprocess import call
        call(['dot','-Tpng','tree.dot','-o','tree.png','-Gdpi=600'])

"""
# Data Collection
X = [ [180, 15,0],     
      [177, 42,0],
      [136, 35,1],
      [174, 65,0],
      [141, 28,1]]

Y = ['man', 'woman', 'woman', 'man', 'woman']    

data_feature_names = [ 'height', 'hair length', 'voice pitch' ]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

dot_data = tree.export_graphviz(clf,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('flower.png')
"""