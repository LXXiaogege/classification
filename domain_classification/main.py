import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.info('******把文本存到X[]中')
dir = {'baby': 129, 'car': 410, 'food': 409, 'health': 406, 'legend': 396, 'life': 409, 'love': 158, 'news': 409,
       'science': 409, 'sexual': 38}
data_file_number=0
x=[]
for world_data_name,world_data_number in dir.items():  #遍历返回键值 给word_data_name和word_data_number
    class_content=[]
    text_content=''
    while(data_file_number<world_data_number):
        trained_file=open('../result/'+world_data_name+'/'+str(world_data_number)+'.txt',mode='r',encoding='utf-8')
        for line in trained_file:
            text_content+=trained_file.readline()
        trained_file.close()
        data_file_number = data_file_number + 1
    class_content.append(text_content)
    x.insert(0,class_content)      #插在前面，设置lable时要反过来
    data_file_number=0

logging.info('******tf-idf转换文本******')
x=np.array(x)
vectorizer=TfidfVectorizer()
vectorizer.max_features=32
x=vectorizer.fit_transform(x.ravel())
logging.info('输出要进行训练的矩阵X的X.Shape')
print(x.shape)

logging.info('******SVM---训练模型******')
y=['sexual','science','news','love','life','legend','health','food','car','baby']
clf=svm.SVC()
clf.fit(x,y)

logging.info('******SVM---领域预测******')
pre_content=open('../result/car/14.txt',mode='r',encoding='utf-8')
pre_line=''
pre_list=[]
for line in pre_content:
    pre_line+=pre_content.readline()
pre_list.append(pre_line)
np.array(pre_list)
vectorizer_pre=TfidfVectorizer()
pre_x=vectorizer_pre.fit_transform(pre_list)
print(clf.predict(pre_x))
