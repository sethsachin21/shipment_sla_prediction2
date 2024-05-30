
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    v = [x for x in request.form.values()]
    
    xcol=['quantity','courier_partner_id_22','courier_partner_id_3',
          'courier_partner_id_30218','courier_partner_id_4',
          'courier_partner_id_5','courier_partner_id_55',
          'courier_partner_id_9','account_type_id_135712',
          'account_type_id_2471','account_type_id_2511',
          'account_type_id_2512','account_type_id_2514',
          'account_type_id_2515','account_type_id_2520',
          'account_type_id_33537','account_type_id_34291',
          'account_type_id_34511','account_mode_Air',
          'account_mode_Default','account_mode_Heavy surface',
          'account_mode_Surface','pickup_pin_code_122506',
          'pickup_pin_code_421311','pickup_pin_code_562123',
          'pickup_pin_code_711313','pickup_pin_code_781035']

    df1_temp=pd.DataFrame(columns=xcol)
    df1_temp.loc[0]=0
    
    df1_temp['quantity']=int(v[0])
    df1_temp['courier_partner_id_'+str(v[1])]=1
    df1_temp['account_type_id_'+str(v[2])]=1
    df1_temp['account_mode_'+str(v[3])]=1
    df1_temp['pickup_pin_code_'+str(v[4])]=1

    prediction = model.predict(df1_temp)    
    
    output = np.round(prediction)

    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)
    #output=df1_temp

    #output = round(prediction[0], 2)

    #return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
    return render_template('index.html', prediction_text='Predicted SLA {}'.format(output))

#app.run()

if __name__ == "__main__":
    app.run(debug=True)