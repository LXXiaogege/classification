import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import logging
#进行文本训练，并把训练模型保存到本地
def training():
    logging.info('******把文本存到X[[][][]...]中')
    dir = {'baby': 129, 'car': 410, 'food': 409, 'health': 406, 'legend': 396, 'life': 409, 'love': 158, 'news': 409,
           'science': 409, 'sexual': 38}
    data_file_number = 0
    x = []
    for world_data_name, world_data_number in dir.items():  # 遍历返回键值 给word_data_name和word_data_number
        class_content = []
        text_content = ''
        while (data_file_number < world_data_number):
            trained_file = open('D:\\Program Files\\JetBrains\\py_workspaces\\classification\\result\\' + world_data_name + '\\' + str(world_data_number) + '.txt', mode='r',
                                encoding='utf-8')
            for line in trained_file:
                text_content += trained_file.readline()
            trained_file.close()
            data_file_number = data_file_number + 1
        class_content.append(text_content)
        x.insert(0, class_content)  # 插在前面，设置lable时要反过来
        data_file_number = 0

    logging.info('******tf-idf转换文本******')
    x = np.array(x)
    vectorizer = TfidfVectorizer()  # 将原始文档集合转换为TF-IDF特征矩阵。
    vectorizer.max_features = 26
    x = vectorizer.fit_transform(x.ravel())
    logging.info('输出要进行训练的矩阵X的X.Shape')
    print(x.shape)

    logging.info('******SVM---训练模型******')
    y = ['sexual', 'science', 'news', 'love', 'life', 'legend', 'health', 'food', 'car', 'baby']
    clf = svm.SVC(decision_function_shape='ovr', gamma='scale', probability=True)
    clf.fit(x, y)
    # 把训练结果保存到本地
    joblib.dump(clf, 'train_module')
training()