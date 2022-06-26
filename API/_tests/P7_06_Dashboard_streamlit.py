import pandas as pd
import streamlit as st
import requests

#------------------------------------------------------------------------------
# Preload data here
#------------------------------------------------------------------------------

headers = {"Content-Type": "application/json"}
# API Web
#api_url = 'http://ec2-18-218-21-153.us-east-2.compute.amazonaws.com:5001/api/'
api_url = 'http://localhost:5001/api/'


def client_list():
    api_test = 'client_list/'
    data_json = {}
    response = requests.request(method='POST', headers=headers,
                            url=api_url+api_test, json=data_json
                        )
    if response.status_code != 200:
        print(f'HTTP error: {response.status_code}')
        return []
    else:
        data_json = response.json()
        return data_json['data']

def prediction(id):
    api_test = 'client_score/'
    data_json = {
        'SK_ID_CURR': id,
        }
    response = requests.request(method='POST', headers=headers,
                            url=api_url+api_test, json=data_json
                        )
    if response.status_code != 200:
        print(f'HTTP error: {response.status_code}')
        return []
    else:
        data_json = response.json()
        return data_json['score']

def main():
    client_choice = st.sidebar.selectbox(
        'Select a client: ',
        client_list())

    st.title('Credit Home Prediction')

    revenu_med = st.number_input('Loan amount',
                                 min_value=0, value=150000, step=10000)

    predict_btn = st.button('Prediction')

    if predict_btn:
        pred = None
        SK_ID_CURR = client_choice[0]
        pred = prediction(SK_ID_CURR)
        st.write(f'Score {pred}')
        


if __name__ == '__main__':
    main()
