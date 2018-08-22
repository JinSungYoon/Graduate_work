# reference : https://pythonprogramminglanguage.com/decision-tree-visual-example/

import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import collections
import pandas as pd

for gender in range(2):
    if gender%15==0:
        Data = pd.read_pickle("Male_data")
    else:
        Data = pd.read_pickle("Female_data")
    
    # 남자 테이블 데이터에서 필요없는 columns를 하나씩 지우는 과정
    del Data["순위"]
    del Data["경기수"]
    del Data["승점"]
    del Data["승"]
    del Data["패"]
    del Data["세트득실률"]
    del Data["점수득실률"]
    
    # 테스트 데이터와 트레이팅 데이터를 담을 리스트를 생성한다.
    Test_X = []
    Test_Y = []
    Train_X = []
    Train_Y = []
    
    # 플레이오프 진출 여부 columns와 결정요소들을 분리한다.
    Y = Data["플레이오프진출"]
    del Data["플레이오프진출"]
    X = Data
    
    # 10배수는 test데이터로 나머지는 training 데이터로 넣는다.
    for loop in range(0,len(Data)):
        if loop%9 == 0:
            Test_Y.append(Y[loop])
            Test_X.append(X.ix[loop].values)
        else:
            Train_Y.append(Y[loop])
            Train_X.append(X.ix[loop].values)
    
    data_feature_name = Data.columns
    
    print(data_feature_name)
    
    # training data로 데이터를 분류하는 작업
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(Train_X,Train_Y)
    
    # 그래프를 그리는 로직
    dot_data = tree.export_graphviz(clf,
                                    feature_names=data_feature_name,
                                    out_file=None,
                                    filled=True,
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    
    colors = ('skyblue', 'pink')
    edges = collections.defaultdict(list)
    
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))
    
    for edge in edges:
        edges[edge].sort()    
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])
    if gender%2==0:
        graph.write_svg('Male_playoff.svg')
    else:
        graph.write_svg('Female_playoff.svg')

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