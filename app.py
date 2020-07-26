import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

def scale(data):
    return((data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0)))

def split_time(feat):
    x1 =[]
    for i in feat:
        if ':' in i:
            first, second = i.split(':', 1)
            x1.append(first)
            x1.append(second)
        else:
            j = int(i)
            x1.append(j)
    final_feat = [int(x) for x in x1]
    return final_feat

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model/DT_AdaBoost.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('templates/test.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = split_time(int_features)
    feat_df = pd.DataFrame(final_features, columns=['data'])
    scale_feat = feat_df.apply(scale)
    scale_list = scale_feat['data'].tolist()
    final_features = [np.array(scale_list)]
    print(final_features)
    prediction = model.predict(final_features)
    print(prediction)
    output = prediction[0]
    print(output)
    if output == 0:
        return render_template('templates/test.html', prediction_text='There is no Departure Delay')
    elif output == 1:
        return render_template('templates/test.html', prediction_text='Delay should be in range of 1 to 5 mins')
    elif output == 2:
        return render_template('templates/test.html', prediction_text='Delay should be in range of 6 to 10 mins')
    elif output == 3:
        return render_template('templates/test.html', prediction_text='Delay should be in range of 11 to 20 mins')
    elif output == 4:
        return render_template('templates/test.html', prediction_text='Delay should be in range of 21 to 50 mins')
    elif output == 5:
        return render_template('templates/test.html', prediction_text='Delay should be in range of 51 to 100 mins')
    elif output == 6:
        return render_template('templates/test.html', prediction_text='Delay should be in range of 101 to 200 mins')
    elif output == 7:
        return render_template('templates/test.html', prediction_text='Delay should be in range of 201 to 1000 mins')
    else:
        return render_template('templates/test.html', prediction_text='Delay should be >1000 mins')

if __name__ == "__main__":
    app.run(debug=True)
