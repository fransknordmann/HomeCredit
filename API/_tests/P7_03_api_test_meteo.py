# Run this api with `python myfile.py` and
# call http://localhost:5000/api/meteo/ from web page or app.

import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/api/meteo/")
def meteo():
    df_meteo = pd.read_csv('Sources/meteo.csv')
    data_meteo = df_meteo[['datetimest', 'temp']].values.tolist()
    return jsonify({
        'status': 'ok', 
        'data': data_meteo
        })

if __name__ == "__main__":
    app.run(debug=True)