import jieba

def pre_process(content):
    stoplist = {}.fromkeys([line.strip() for line in open("D:\\Program Files\\JetBrains\\py_workspaces\\classification\\domain_classification\\stopwords.txt", encoding='UTF-8')])
    #分词
    seg_list = jieba.lcut(content, cut_all=False)
    # 去除停用词
    seg_list = [word for word in list(seg_list) if word not in stoplist]
    output_file=open('pre_test_data.txt',mode='w',encoding='utf-8')
    for i in range(len(seg_list)):
        output_file.write(str(seg_list[i])+'\n')