# Data Scientist - P7 - Laurent Trichet
# Implémentez un modèle de scoring
# 
# API Web
# Run this api with `python myfile.py` and
# e.g. call http://localhost:5001/api/client_list/ from web page or app.
# to log info : app.logger.info('my message')

# Import standard librairies
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import requests

# Import Garbage Collector (empty dataFrame memory)
import gc

app = Flask(__name__)


#------------------------------------------------------------------------------
# Preload data here
#------------------------------------------------------------------------------

# Constants
DIRDATASET = './credithome_datasets/'
NUMROWS = 15000       # 1000000 = total dataset
FILESTD_FNAN0_REDUCED =\
    DIRDATASET+'Credit_Home_Junction_Std_Fnan0_Reduced_'+str(NUMROWS)+'.csv'
FILEFNR_IN = DIRDATASET+'Credit_Home_FalseNegativeRates.csv'
FILEFPR_IN = DIRDATASET+'Credit_Home_FalsePositiveRates.csv'
FILEFEAT_IN = DIRDATASET+'Credit_Home_Features.csv'
FILEFILT_IN = DIRDATASET+'Credit_Home_Filters.csv'

# load data training and data test
df = pd.read_csv(FILESTD_FNAN0_REDUCED, sep='\t')
s = df.select_dtypes(include='int').columns
df[s] = df[s].astype("float")

# load threshold of false negative and positive rates
df_fnr = pd.read_csv(FILEFNR_IN, sep='\t')
df_fpr = pd.read_csv(FILEFPR_IN, sep='\t')

# load features definitions and values and keep 12 main values
df_feat = pd.read_csv(FILEFEAT_IN, sep='\t')
df_feat = df_feat.iloc[0:12,:]

# load filters
df_filter = pd.read_csv(FILEFILT_IN, sep='\t')

# list of features
key_features = ['SK_ID_CURR', 'CODE_GENDER', 'AMT_CREDIT', 'DAYS_BIRTH']
main_features = df_feat['col name'].values.tolist()

def map_sk_id_curr(c):
    return str(c)

def map_code_gender(c):
    if c==1:
        return "Man"
    else:
        return "Woman"

def map_amt_credit(c):
    return 'USD $'+str(c)

def map_days_birth(c):
    return str(c)+' years old'

# prepare description for client_list (dropdown menu)
data_cli = df[df['TARGET']==999][key_features]
data_cli['SK_ID_CURR'] = data_cli['SK_ID_CURR'].astype('int')
data_cli['SK_ID_CURR'] = data_cli['SK_ID_CURR'].astype('object')
data_cli['SK_ID_CURR'] = data_cli['SK_ID_CURR'].map(map_sk_id_curr)
data_cli['CODE_GENDER'] = data_cli['CODE_GENDER'].astype('object')
data_cli['CODE_GENDER'] = data_cli['CODE_GENDER'].map(map_code_gender)
data_cli['AMT_CREDIT'] = data_cli['AMT_CREDIT'].astype('int')
data_cli['AMT_CREDIT'] = data_cli['AMT_CREDIT'].astype('object')
data_cli['AMT_CREDIT'] = data_cli['AMT_CREDIT'].map(map_amt_credit)
data_cli['DAYS_BIRTH'] = -data_cli['DAYS_BIRTH']
data_cli['DAYS_BIRTH'] = data_cli['DAYS_BIRTH'].astype('int')
data_cli['DAYS_BIRTH'] = data_cli['DAYS_BIRTH'].astype('object')
data_cli['DAYS_BIRTH'] = data_cli['DAYS_BIRTH'].map(map_days_birth)
data_cli['DESC'] = data_cli['SK_ID_CURR'] + ' - ' +\
                   data_cli['CODE_GENDER'] + ' - ' +\
                   data_cli['DAYS_BIRTH'] + ' - Amount Loan: ' +\
                   data_cli['AMT_CREDIT']

# prepare data for client_score
model_url = 'http://localhost:5000/invocations'
#model_url = 'http://oc.14eight.com:5000/invocations'
headers = {"Content-Type": "application/json"}
c_features = [c for c in df.columns \
              if c not in ['index', 'TARGET', 'SK_ID_CURR']]
df_test = df[df['TARGET']==999]


#------------------------------------------------------------------------------
# APIs
#------------------------------------------------------------------------------

@app.route("/api/client_list/", methods=['POST'])
def client_list():
    """
    Return complete list of clients (loans) for the dataset test. 
    
    Parameters
    ----------
    None

    Returns
    -------
    status : OK if correct request
    data: list of clients with features
          'SK_ID_CURR', 'CODE_GENDER', 'AMT_CREDIT', 'DAYS_BIRTH'
          DAYS_BIRTH converted to strings with feature description 
    
    """
    data = data_cli.iloc[135:160,:]['DESC'].values.tolist()

    return jsonify({
        'status': 'ok', 
        'data': data
        })


@app.route("/api/client_score/", methods=['POST'])
def client_score():
    """
    Return score =  probability to be positive. 
    
    Parameters
    ----------
    SK_ID_CURR : id of client (loan)

    Returns
    -------
    status : OK if correct request
    score: value between 0.0 and 1.0 
    
    """
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        id_client = json['SK_ID_CURR']
        data_client = df_test[df_test['SK_ID_CURR']==int(id_client)]\
                     [c_features].iloc[0,:].to_list()
        data_json = {'data': [data_client]}
        response = requests.request(method='POST',
                            headers=headers,
                            url=model_url,
                            json=data_json
                        )
        if response.status_code != 200:
            return jsonify({
                'status': 'ok', 
                'score': response.json(),
                })
        else:
            return jsonify({
                'status': 'ok', 
                'score': response.json(),
                })
    else:
        return f'Content-Type {content_type} not supported'


@app.route("/api/feature_list/", methods=['POST'])
def feature_list():
    """
    Return list of 12 main features to use in dashboard. Use the 12 most
    important features. Could be extended to all features in a complete
    dashboard. 
    
    Parameters
    ----------
    None

    Returns
    -------
    status: OK if correct request
    data: list of 12 features with
          'col name', 'mean importance', 'description'
    """

    data_feat = df_feat[['col name','mean importance', 'description']]
    data = data_feat.values.tolist()
    del data_feat
    gc.collect()

    return jsonify({
        'status': 'ok', 
        'data': data
        })



@app.route("/api/fnr_list/", methods=['POST'])
def fnr_list():
    """
    Return list of False Negative Rates and threshold recorded during
    the modelisation phase 
    
    Parameters
    ----------
    None

    Returns
    -------
    status: OK if correct request
    data: list of False Negative Rate + Threshold
    """

    data = df_fnr.values.tolist()

    return jsonify({
        'status': 'ok', 
        'data': data
        })



@app.route("/api/fpr_list/", methods=['POST'])
def fpr_list():
    """
    Return list of False Negative Rates and threshold recorded during
    the modelisation phase 
    
    Parameters
    ----------
    None

    Returns
    -------
    status: OK if correct request
    data: list of False Negative Rate + Threshold
    """

    data = df_fpr.values.tolist()

    return jsonify({
        'status': 'ok', 
        'data': data
        })



@app.route("/api/feature_values/", methods=['POST'])
def feature_values():
    """
    Return list of 12 main features values for a client and list of min,
    mean, max values of the 12 features for negative and positive clients
    of the whole training dataset  (if no filter) or of the filtered
    dataset on ranges delivered in entry
    
    Parameters
    ----------
    SK_ID_CURR : id of client (loan)
    FILTER_Y_N : Y (use ranges), N (use whole training dataset)
    RANGES : list of 12 [min, max] of the 12 features

    Returns
    -------
    status : OK if correct request
    data_client : list of values for client SK_ID_CURR
    population : list with count positive pop and negative pop
    data : list of 12 feature with 'col name',
           'min positive', 'mean positive', 'max positive',
           'min negative', 'mean negative', 'max negative'
    
    """

    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        id_client = json['SK_ID_CURR']
        data_client = df[df['SK_ID_CURR']==int(id_client)]\
                        [main_features].values.tolist()
        for i in np.arange(0, len(data_client[0])):
            data_client[0][i] = round(100*data_client[0][i])/100
        i_filter = json['FILTER_Y_N']
        df_pos = df[df['TARGET']==1][main_features]
        df_neg = df[df['TARGET']==0][main_features]
        if i_filter == 'Y':
            ranges = json['RANGES']
            # search mins et maxs in full dataset
            for f_i in np.arange(0, len(ranges)):
                f_name = df_feat.iloc[f_i,:]['col name']
                df_pos = df_pos[df_pos[f_name]>=float(ranges[f_i][0])]
                df_pos = df_pos[df_pos[f_name]<=float(ranges[f_i][1])]
                df_neg = df_neg[df_neg[f_name]>=float(ranges[f_i][0])]
                df_neg = df_neg[df_neg[f_name]<=float(ranges[f_i][1])]
            pop_pos = df_pos.shape[0]
            pop_neg = df_neg.shape[0]
            data = []
            for f_i in np.arange(0, len(ranges)):
                f_name = df_feat.iloc[f_i,:]['col name']
                data.append(
                        [
                        f_name,
                        data_client[0][f_i],
                        round(100*df_pos[f_name].min())/100,
                        round(100*df_pos[f_name].mean())/100,
                        round(100*df_pos[f_name].max())/100,
                        round(100*df_neg[f_name].min())/100,
                        round(100*df_neg[f_name].mean())/100,
                        round(100*df_neg[f_name].max())/100,
                        ]
                    )
        else:
            # min, mean, max already calculated in df_feat
            pop_pos = df_pos.shape[0]
            pop_neg = df_neg.shape[0]
            df_feat['val'] = data_client[0]
            app.logger.info(df_feat['val'])
            data = df_feat[
                ['col name', 'val',
                 'min positive', 'mean positive', 'max positive',
                 'min negative', 'mean negative', 'max negative'
                ]
            ].values.tolist()
        del df_pos, df_neg
        gc.collect()

        return jsonify({
            'status': 'ok', 
            'data_client': data_client,
            'population': [pop_pos, pop_neg],
            'data': data
            })
    else:
        return f'Content-Type {content_type} not supported'



@app.route("/api/population_count/", methods=['POST'])
def population_count():
    """
    Return count positive and negative clients for the filtered
    dataset on ranges delivered in entry
    
    Parameters
    ----------
    FILTER_Y_N : Y (use ranges), N (use whole training dataset)
    RANGES : list of 12 [min, max] of the 12 features

    Returns
    -------
    status : OK if correct request
    data_client : list of values for client SK_ID_CURR
    population : list with count positive pop and negative pop
    
    """

    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        i_filter = json['FILTER_Y_N']
        df_pos = df[df['TARGET']==1][main_features]
        df_neg = df[df['TARGET']==0][main_features]
        if i_filter == 'Y':
            ranges = json['RANGES']
            # search mins et maxs in full dataset
            for f_i in np.arange(0, len(ranges)):
                f_name = df_feat.iloc[f_i,:]['col name']
                df_pos = df_pos[df_pos[f_name]>=float(ranges[f_i][0])]
                df_pos = df_pos[df_pos[f_name]<=float(ranges[f_i][1])]
                df_neg = df_neg[df_neg[f_name]>=float(ranges[f_i][0])]
                df_neg = df_neg[df_neg[f_name]<=float(ranges[f_i][1])]
            pop_pos = df_pos.shape[0]
            pop_neg = df_neg.shape[0]
        else:
            # min, mean, max already calculated in df_feat
            pop_pos = df_pos.shape[0]
            pop_neg = df_neg.shape[0]
        del df_pos, df_neg
        gc.collect()

        return jsonify({
            'status': 'ok', 
            'population': [pop_pos, pop_neg],
            })
    else:
        return f'Content-Type {content_type} not supported'



@app.route("/api/filters/", methods=['POST'])
def filters_values():
    """
    Return list of ranges of filters for 12 features
    
    Parameters
    ----------
    None

    Returns
    -------
    list of ranges with col name, rank, min, max, description
    
    """
    
    data_filter = df_filter.values.tolist()

    return jsonify({
        'status': 'ok', 
        'data': data_filter
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)