from flask import Flask,request
from domain_classification import testData_process,classify_dataPre

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def domainClassify():
    content=request.form.get('content')
    testData_process.pre_process(content)  # 预处理
    json_data=classify_dataPre.pre_data()
    return json_data

if __name__ =='__main__':
    app.run(debug=True)