from flask import Flask,jsonify,request
from domain_classification import testData_process,classify_dataPre

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def domainClassify():
    content=request.form.get('content')
    testData_process.pre_process(content)  # 预处理
    string=str(classify_dataPre.pre_data().tolist())    #把numpy.ndarray转化为list再转化为str
    return string

if __name__ =='__main__':
    app.run(debug=True)