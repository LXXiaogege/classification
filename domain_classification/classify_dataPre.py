import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import json

#用本地保存的模型进行测试，输出每个领域的概率，和概率最大的领域，

def pre_data():
    pre_content=open('D:\\Program Files\\JetBrains\\py_workspaces\\classification\\controller\\pre_test_data.txt',mode='r',encoding='utf-8')
    pre_line = ''
    pre_list = []
    for line in pre_content:
        pre_line += pre_content.readline()
    pre_list.append(pre_line)
    pre_list=np.array(pre_list)
    vectorizer_pre = TfidfVectorizer()
    vectorizer_pre.max_features=26
    pre_x = vectorizer_pre.fit_transform(pre_list.ravel())
    clf = joblib.load('D:\\Program Files\\JetBrains\\py_workspaces\\classification\\domain_classification\\train_module')  # 模型本地回调
    # print(clf.predict(pre_x)  )              #概率最大
    # print(clf.predict_proba(pre_x))          #概率
    # print(clf.predict_log_proba(pre_x))      #分类概率的对数

    proba = clf.predict_proba(pre_x).tolist()  # 把numpy.ndarray转化为list再转化为str
    dir = {'baby': proba[0][9], 'car': proba[0][8], 'food': proba[0][7], 'health': proba[0][6], 'legend': proba[0][5],
           'life': proba[0][4], 'love': proba[0][3], 'news': proba[0][2],
           'science': proba[0][1], 'sexual': proba[0][0]}
    json_data = json.dumps(dir)
    return json_data
pre_data()
