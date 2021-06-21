from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json
import pandas as pd
from fbprophet import Prophet

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/": {"origins": "*"}})

@app.route("/", methods=["POST"])
@cross_origin(origin='*',headers=['Content- Type','Authorization'])

def index():
    data = json.loads(request.data.decode())['data']
    df = pd.DataFrame(data[1:], columns = data[0])
    df.head()
    df.describe()
    df = df[["Date","Open"]]
    df = df.rename(columns={"Date":"ds","Open":"y" })
    df.head()
    m = Prophet(yearly_seasonality = "auto", weekly_seasonality= "auto", daily_seasonality = "auto")
    m.fit(df)
    future = m.make_future_dataframe(periods = 90, include_history = False)
    prediction = m.predict(future)
    prediction = prediction.rename(columns = {"ds":"Date"})
    prediction = prediction[['Date', 'trend', 'trend_lower', 'trend_upper']]
    csvData = prediction.to_csv()
    return jsonify({'data': csvData})