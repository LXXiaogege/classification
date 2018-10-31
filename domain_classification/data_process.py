import jieba

dir = {'baby': 129, 'car': 410, 'food': 409, 'health': 406, 'legend': 396, 'life': 409, 'love': 158, 'news': 409,
       'science': 409, 'sexual': 38}
# 设置词典，分别是类别名称和该类别下一共包含的文本数量
data_file_number = 0
# 当前处理文件索引数

for world_data_name, world_data_number in dir.items():
    # 将词典中的数据分别复制到world_data_name,world_data_number中
    while (data_file_number < world_data_number):
        # 打印文件索引信息
        print(world_data_name)
        print(world_data_number)
        print(data_file_number)
        file = open('../training_data/' + world_data_name + '/' + str(data_file_number) + '.txt',mode='r', encoding='UTF-8')
        file_w = open('../result/' + world_data_name + '/' + str(data_file_number) + '.txt', mode='w', encoding='UTF-8')
        for line in file:
            stoplist = {}.fromkeys([line.strip() for line in open("stopwords.txt", encoding='UTF-8')])
            seg_list = jieba.lcut(line, cut_all=False)
            # 去除停用词
            seg_list = [word for word in list(seg_list) if word not in stoplist]
            print("Default Mode:", "/ ".join(seg_list))
            for i in range(len(seg_list)):
                file_w.write(str(seg_list[i]) + '\n')     #一行一行写入
        file_w.close()
        file.close()
        data_file_number = data_file_number + 1
    data_file_number = 0
