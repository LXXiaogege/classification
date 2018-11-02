import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

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
    print(clf.predict_proba(pre_x))   #概率
    print(clf.predict_log_proba)      #分类概率的对数
    return clf.predict(pre_x)         #概率最大
