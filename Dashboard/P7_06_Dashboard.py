# Data Scientist - P7 - Laurent Trichet
# Implémentez un modèle de scoring
# 
# Dashboard
#
from dash import Dash, dcc, html, Input, Output, dash_table
import dash._callback_context as ctx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests

app = Dash(__name__)


_TRACE_TXT = False
def trace_txt(message):
    if _TRACE_TXT:
        outf = open('trace.txt', 'a' )
        outf.write(f'{message}\n')
        outf.close()

#------------------------------------------------------------------------------
# Preload data here
#------------------------------------------------------------------------------

# API Web
api_url = 'http://oc.14eight.com:5001/api//'
headers = {'Content-Type': 'application/json'}

def prediction(id):
    # call API to obtain score of selected item
    api_test = 'client_score/'
    data_json = {'SK_ID_CURR': id}
    response = requests.request(method='POST', headers=headers,
                            url=api_url+api_test, json=data_json
                        )
    if response.status_code != 200:
        print(f'HTTP error: {response.status_code}')
        return []
    else:
        data_json = response.json()
        return data_json['score'][0]

def feature_values(id, filter, data_filter):
    # call API to obtain values of selected item for main features
    api_test = 'feature_values/'
    data_json = {
        'SK_ID_CURR': id,
        'FILTER_Y_N': filter,
        'RANGES': data_filter,
        }
    response = requests.request(method='POST', headers=headers,
                            url=api_url+api_test, json=data_json
                        )
    if response.status_code != 200:
        print(f'HTTP error: {response.status_code}')
        return []
    else:
        data_json = response.json()
        pop = data_json['population']
        data = data_json['data']
        return pop, data

def f_list(name_list):
    # generifc call API to obtain different lists
    api_test = name_list
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

def list_match_slider(inlist):
    # Maintain number of marks <= 7 in slider for visual improvt.
    outlist = []
    for i in np.arange(len(inlist)):
        if inlist[i][0] in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            outlist.append(inlist[i])
    return outlist

def calc_slider_marks(falserate_list, minlab, maxlab):
    # position marker in slider according to proba of thresholds
    slider_marks = {0:minlab}
    for i in np.arange(0, len(falserate_list)):
        percent = f'{int(100*falserate_list[i][0])}'
        mark = 100*falserate_list[i][1]
        slider_marks[int(mark)] = percent
    slider_marks[100] = maxlab
    return slider_marks

def update_slider_marks(score, slider_marks, falserate_list):
    # insert mark for score into slider
    i=0
    while i < len(falserate_list):
        if score > falserate_list[i][1]:
            break
        else:
            i = i + 1
    i = min(i, len(falserate_list)-1)
    left_risk_mark = 100*falserate_list[i][1]
    falserate_thr = slider_marks[int(left_risk_mark)]
    if i > 0:
        right_risk_mark = 100*falserate_list[i-1][1]
    else:
        right_risk_mark = 100
    if score > 0 and score < 1:      
        slider_marks[100*score] = ''      # add score mark
    return falserate_thr, left_risk_mark, right_risk_mark, slider_marks

def calc_gauge_steps(falserate_list, colors_ranges):
    # calc gauge steps (colors) according to threshold proba
    steps_gauge = [{'range': [falserate_list[0][1], 1],
                        'color': colors_ranges[0]}]
    for i in np.arange(0, len(falserate_list)-1):
        steps_gauge.append(
            {'range': [falserate_list[i+1][1], falserate_list[i][1]],
            'color': colors_ranges[i+1]})
    steps_gauge.append(
        {'range': [0, falserate_list[6][1]],
        'color': colors_ranges[7]})
    return steps_gauge

def map_values_to_0_1(df_feat_val, tab_f):
    # map all features values to a range 0 to 1 (align graphics)
    y1 = []
    y2 = []
    y3 = []
    for i in np.arange(0, df_feat_val.shape[0]):
        ft = df_feat_val.iloc[i,:]['Feature']
        lv = df_feat_val.iloc[i,:]['Loan value']
        mp =  df_feat_val.iloc[i,:]['mean positive']
        mn =  df_feat_val.iloc[i,:]['mean negative']
        tf = tab_f[i][1]
        tf = tf[tf['Feature']==ft]
        if lv >= 0:
            max_feat = tf[tf['rank']==0]['max'].values[0]
        else:
            max_feat = 0-tf[tf['rank']==0]['min'].values[0]
            lv = 0-lv
            mp = 0-mp
            mn = 0-mn
        lv = lv / max_feat
        mp = mp / max_feat
        mn = mn / max_feat
        y1.append(mp)
        y2.append(lv)
        y3.append(mn)
    return y1, y2, y3

def get_recommendation(score):
    label_recommendation = [
'Credit Home recommends to reject this loan, the risk of unpayment is too high.\
 Feature importance and values below can help clarify the score.',
'Credit Home recommends that you discuss the decision with your hierarchy.\
 Feature importance and values below can help clarify the score.',
'Credit Home recommends to accept this loan, the risk is low.\
 Feature importance and values below can help clarify the score.']
    if score > 0.72:
        return label_recommendation[2], 'traffic_light_green.png'
    else:
        if score > 0.5:
            return label_recommendation[1], 'traffic_light_orange.png'
        else:
            return label_recommendation[0], 'traffic_light_red.png'

#------------------------------------------------------------------------------
# Styles & text
#------------------------------------------------------------------------------

colors = {
    'background': '#FFDDDD',
    'text': '#444444'
}
colors_ranges_risk = ['#FFEFEF', '#FFCFCF', '#FF9F9F', '#FF7F7F',
                      '#FF5F5F', '#FF3F3F', '#FF1F1F', '#FF0F0F']
colors_ranges_turnover = ['#729AAC', '#8BBCD2', '#A4DEF8', '#BCE5FF', 
                          '#C7E9FD', '#CEECFF', '#D8EFFF', '#F3FAFF']

label_title = 'CREDIT HOME > loan requests validation'
label_dropdown = 'Please select a client loan request:'
label_risk_beg = 'The risk to accept a loan that would not be repaid is around '
label_risk_end = '% (False positive)'
label_gauge_risk = 'Score & risk estimation'
label_ranges_risk = 'False Positive Ranges:'
label_turnover_beg = 'The probability to refuse a loan that would have been repaid is greater than '
label_turnover_end = '% (False negative)'
label_gauge_turnover = 'Score & potential loss of earnings'
label_ranges_turnover = 'False Negative Ranges:'
label_feature_importance = 'Features Importance'
label_feature_values = 'Values for this loan compared to the values of past contracts, '

font_size_title = '20px'
font_size_label_dropdown = '16px'
font_size_loan_dropdown = '14px'
font_size_gauge_title = 16
font_size_gauge_label = '18px'
font_size_label_recommendation = '18px'
font_size_feature_importance = '12px'
font_size_feature_values = '12px'

#------------------------------------------------------------------------------
# Preparation of main components items
#------------------------------------------------------------------------------

# LOANS : list for dropdown menu
clients_of_the_day = f_list('client_list/')

# FEATURES : list of main features and their importance
features_list = pd.DataFrame(f_list('feature_list'),
                   columns=['Feature', 'Weight', 'Description'])

# FEATURE VALUES : preparation for table of values
feat_values_columns = ['Feature', 'Loan value',
                       'min pos.', 'mean positive', 'max pos.',
                       'min neg.', 'mean negative', 'max neg.']

# FILTERS: tab with name of feature, dataFrame of options, and value selected 
feat_filters = pd.DataFrame(f_list('filters/'),
                  columns=['Feature', 'rank', 'min', 'max', 'description'])
tab_filters = []
for i in np.arange(0, features_list.shape[0]):
    f_name = features_list.iloc[i,:]['Feature']
    tab_filters.append([f_name,
                        feat_filters[feat_filters['Feature']==f_name]])

# RANGES: risks (false positive) and turnover loss (false negative)

# Calc marks positions in slider for ranges of false positives
risk_fpr_list = f_list('fpr_list/')
risk_fpr_list = list_match_slider(risk_fpr_list)
risk_marks = calc_slider_marks(risk_fpr_list, '100%', '0%')
# Calc steps colors for ranges of false positives
steps_gauge_risk = calc_gauge_steps(risk_fpr_list, colors_ranges_risk)

# Calc marks positions in slider for ranges of false negatives
turnover_fnr_list = f_list('fnr_list')
turnover_fnr_list = list_match_slider(turnover_fnr_list)
turnover_marks = calc_slider_marks(turnover_fnr_list, '0%', '100%')
# Calc steps colors for ranges of false negatives
steps_gauge_turnover = calc_gauge_steps(turnover_fnr_list,
                                        colors_ranges_turnover)


#------------------------------------------------------------------------------
# Dash HTML Components
#------------------------------------------------------------------------------

app.layout = html.Div([
    html.Div(children=[    # HEADER SECTION
        html.Label(id='label_title',
            children=label_title,
            style={'textAlign': 'left', 'font-family': 'Arial','margin-left': '12px',
                'font-size': font_size_title, 'color': '#729AAC'}
        ),
        html.Img(src=app.get_asset_url('cyan_line.png'),
            width='100%', height='2px'
        ),
    ], style={'backgroundColor':'#FFFFFF'}),

    html.Div(children=[    # CONTENT SECTION
        html.Div(children=[    # CONTENT Dropdown loans, score and decision
            html.Div(children=[    # Label dropdown loans
                html.Label(id='label_dropdown',
                    children=label_dropdown,
                    style={'textAlign': 'left', 'font-family': 'Arial','margin-left': '6px',
                        'font-size': font_size_label_dropdown, 'verticalAlign':'middle'}
                ),
            ], style={'display': 'inline-block', 'verticalAlign':'middle'}),
            html.Div(children=[    # dropdown loans
                dcc.Dropdown(
                    clients_of_the_day,
                    clients_of_the_day[0],
                    id='client-dropdown',
                    style={'height':'24','width': '90%','align':'left','margin-top': '0px',
                        'margin-left': '4px','font-family': 'Arial',
                        'font-size': font_size_loan_dropdown}
                ),
            ], style={'display': 'inline-block', 'width': '65%', 'text-align':'left', 'verticalAlign':'middle'}),
            html.Img(src=app.get_asset_url('cyan_line.png'),
                width='100%', height='2px'
            ),
            html.Div(children=[    # Gauges and Recommendation
                html.Div(children=[    # Gauge and slider risk
                    dcc.Graph(id='gauge-risk'
                    ),
                    html.Div(children=[
                        html.Label(id='label-risk',
                            children=label_risk_beg,
                            style={'textAlign': 'center', 'font-family': 'Arial',
                            'font-size': font_size_gauge_label}
                        ),
                        html.Br(), html.Br(), html.Br(), 
                        html.Label(id='label-ranges-risk',
                            children=label_ranges_risk,
                            style={'textAlign': 'center', 'font-family': 'Arial',
                            'font-size': font_size_gauge_label}
                        ),
                        html.Br(), html.Br(),
                        dcc.RangeSlider(id='slider-risk',
                            min=0, max=100, step=None,
                            marks=risk_marks,
                            value=[0,0],
                        ),
                    ], style={'width': '90%', 'text-align':'center','padding-left':'5%', 'padding-right':'5%'})
                ], style={'display': 'inline-block', 'width': '50%'}),
                html.Div(children=[    # Gauge and slider Turnover
                    dcc.Graph(id='gauge-turnover'
                    ),
                    html.Div(children=[
                        html.Label(id='label-turnover',
                            children=label_turnover_beg,
                            style={'textAlign': 'center', 'font-family': 'Arial',
                            'font-size': font_size_gauge_label}
                        ),
                        html.Br(), html.Br(), html.Br(), 
                        html.Label(id='label-ranges-turnover',
                            children=label_ranges_turnover,
                            style={'textAlign': 'center', 'font-family': 'Arial',
                            'font-size': font_size_gauge_label}
                        ),
                        html.Br(), html.Br(),
                        dcc.RangeSlider(id='slider-turnover',
                            min=0, max=100, step=None,
                            marks=turnover_marks,
                            value=[0,0],
                        ),
                    ], style={'width': '90%', 'text-align':'center','padding-left':'5%', 'padding-right':'5%'}),
                ], style={'display': 'inline-block', 'width': '50%' }),
                html.Br(), html.Br(),
                html.Img(src=app.get_asset_url('cyan_line.png'),
                    width='100%', height='2px'
                ),
                html.Br(),
                html.Div(children=[    # Decision and recommendation padding
                ], style={'display': 'inline-block', 'width': '1%'}),
                html.Div(children=[    # Decision and recommendation image
                    html.Img(id='image-recommendation',
                        src=app.get_asset_url('traffic_light_orange.png'),
                        width='52px', height='108px'
                    ),
                ], style={'display': 'inline-block', 'width': '10%', 'vertical-align':'baseline'}),
                html.Div(children=[    # Decision and recommendation text
                    html.Br(),
                    html.Label(id='title-recommendation',
                        children='RECOMMENDATION:',
                        style={'textAlign': 'left', 'font-family': 'Arial',
                               'font-size': font_size_title, 'color': '#729AAC'}
                    ),
                    html.Br(), html.Br(),
                    html.Label(id='label-recommendation',
                        children='',
                        style={'textAlign': 'left', 'font-family': 'Arial',
                               'font-size': font_size_label_recommendation}
                    ),
                ], style={'display': 'inline-block', 'width': '87%', 'text-align':'left', 'vertical-align':'top'}),
                html.Img(src=app.get_asset_url('cyan_line.png'),
                    width='100%', height='2px'
                ),
            ], style={'width': '100%', 'text-align':'center'}),
        ], style={'padding': 5, 'display': 'inline-block', 'width': '85%' }),

        html.Div(children=[    # CONTENT Features
            html.Div(children=[     # Feature Importance
                html.Br(),
                html.Label(id='label-feature-importance',
                    children=label_feature_importance,
                    style={'textAlign': 'center', 'font-family': 'Arial','margin-left': '6px',
                        'font-size': '18px'}
                ),
                html.Br(), html.Br(),
                dash_table.DataTable(id='feature-importance',
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'fontSize': font_size_feature_importance,
                    },
                    style_as_list_view=True,
                    style_cell={'padding': '4px'},
                    style_header={
                        'backgroundColor': '#A9DFF8',
                        'fontWeight': 'bold',
                        'fontSize': font_size_feature_importance,
                    },
                    style_cell_conditional=[
                        {'if': {'column_id': 'Feature'},
                            'textAlign': 'left',
                            'fontWeight': 'bold',
                        },
                        {'if': {'column_id': 'Weight'},
                            'textAlign': 'center',
                            'fontWeight': 'bold',
                        },
                        {'if': {'column_id': 'Description'},
                            'textAlign': 'left',
                        },
                    ],
                    data=features_list.to_dict('records'),
                    columns=[{'id': c, 'name': c} for c in features_list.columns],
                    page_size=6
                ),
                html.Img(src=app.get_asset_url('cyan_line.png'),
                    width='100%', height='2px'
                ),
            ], style={'width': '100%', 'textAlign': 'left'}),
            html.Div(children=[     # Feature Values
                html.Br(), html.Br(),
                html.Label(id='label-feature-values',
                    children=label_feature_values,
                    style={'textAlign': 'center', 'font-family': 'Arial','margin-left': '6px',
                        'font-size': '18px'}
                ),
                html.Br(), html.Br(),
                dash_table.DataTable(id='feature-values',
                    style_data={
                        'fontSize': font_size_feature_values,
                    },
                    style_as_list_view=True,
                    style_cell={'padding': '4px'},
                    style_header={
                        'backgroundColor': '#A9DFF8',
                        'fontSize': font_size_feature_values,
                        'fontWeight': 'bold',
                    },
                    style_cell_conditional=[
                        {'if': {'column_id': 'Feature'},
                            'textAlign': 'left',
                            'fontWeight': 'bold',
                            'color': 'black',
                        },
                        {'if': {'column_id': 'Loan value'},
                            'fontWeight': 'bold',
                            'color': 'black',
                        },
                        {'if': {'column_id': 'min pos.'},
                            'color': '#999999',
                        },
                        {'if': {'column_id': 'mean positive'},
                            'fontWeight': 'bold',
                            'color': 'green'
                        },
                        {'if': {'column_id': 'max pos.'},
                            'color': '#999999',
                        },
                        {'if': {'column_id': 'min neg.'},
                            'color': '#999999',
                        },
                        {'if': {'column_id': 'mean negative'},
                            'fontWeight': 'bold',
                            'color': 'orange'
                        },
                        {'if': {'column_id': 'max neg.'},
                            'color': '#999999',
                        },
                    ],
                    data=[{}],
                    columns=[{'id': c, 'name': c} for c in feat_values_columns],
                    page_size=12
                ),
                html.Br(),
            ], style={'width': '100%', 'textAlign': 'left'}),
            html.Div(children=[     # Filters
                html.Label(id='label-filters',
                    children='Filters',
                    style={'textAlign': 'center', 'font-family': 'Arial','margin-left': '6px',
                        'font-size': '18px'}
                ),
                html.Br(),
                html.Table([
                    html.Tr([
                        html.Td(
                            dcc.Dropdown(
                                tab_filters[4*lig_feat+col_feat][1]['description'].values,
                                placeholder=tab_filters[4*lig_feat+col_feat][0],
                                id=f'filter-{4*lig_feat+col_feat}-dropdown',
                                style={'height':'6','width': '250px','align':'left',
                                    'margin-top': '0px', 'margin-left': '2px', 
                                    'font-family': 'Arial','font-size': '11px', 'color': 'black'}
                            ),
                        ) for col_feat in np.arange(0, 4)])
                for lig_feat in np.arange(0,3)]),
            ], style={'padding': 18, 'width': '90%', 'textAlign': 'left'}),
            html.Div(children=[     # Graphic
                dcc.Graph(id='feature-graphic'
                ),
            ], style={'width': '100%', 'textAlign': 'left'}),
        ], style={'padding': 5, 'display': 'inline-block', 'width': '85%' })
    ], style={'backgroundColor':'#FFFFFF'}),

], style={'backgroundColor':'#FFFFFF'})

#------------------------------------------------------------------------------
# Dash Callbacks
#------------------------------------------------------------------------------

@app.callback(
    Output('feature-graphic', 'figure'),
    Output('label-feature-values', 'children'),
    Output('feature-values', 'data'),
    Output('label-risk', 'children'),
    Output('gauge-risk', 'figure'),
    Output('slider-risk', 'marks'),
    Output('slider-risk', 'value'),
    Output('label-turnover', 'children'),
    Output('gauge-turnover', 'figure'),
    Output('slider-turnover', 'marks'),
    Output('slider-turnover', 'value'),
    Output('image-recommendation', 'src'),
    Output('label-recommendation', 'children'),
    Input('filter-0-dropdown', 'value'),
    Input('filter-1-dropdown', 'value'),
    Input('filter-2-dropdown', 'value'),
    Input('filter-3-dropdown', 'value'),
    Input('filter-4-dropdown', 'value'),
    Input('filter-5-dropdown', 'value'),
    Input('filter-6-dropdown', 'value'),
    Input('filter-7-dropdown', 'value'),
    Input('filter-8-dropdown', 'value'),
    Input('filter-9-dropdown', 'value'),
    Input('filter-10-dropdown', 'value'),
    Input('filter-11-dropdown', 'value'),
    Input('client-dropdown', 'value')
)
def update_client_dropdown(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,client):

    # collect parameters from callback context (multi Input fields)
    
    SK_ID_CURR = ctx.callback_context.inputs['client-dropdown.value'][0:6]
    # Search for description of selected filters in tab_filter
    filter_list = []
    for i_feat in np.arange(0, len(tab_filters)):
        choice_feat = ctx.callback_context.inputs[
                          f'filter-{i_feat}-dropdown.value']
        rank, tbf = 0, tab_filters[i_feat][1]
        for i in np.arange(0, len(tbf)):
            if tbf.iloc[i,:]['description'] == choice_feat:
                rank = i
                break
        min_max = [tbf.iloc[rank,:]['min'], tbf.iloc[rank,:]['max']]
        filter_list.append(min_max)

    # call APIs to get score and feature values
    score = round(100*prediction(SK_ID_CURR))/100
    population, feat_values = feature_values(SK_ID_CURR, 'Y', filter_list)
    lb_feat_values = f'{label_feature_values}{population[0]} positive loans and {population[1]} negative loans'
    df_feat_values = pd.DataFrame(feat_values, columns= feat_values_columns)

    # graphic version of feature values
    fig_feat = go.Figure()
    fig_feat.layout.paper_bgcolor = "#F3FAFF"
    fig_feat.update_layout(title='Relative positions')
    fig_feat.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig_feat.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    y1, y2, y3 = map_values_to_0_1(df_feat_values, tab_filters)
    fig_feat.add_trace(go.Scatter(
                x = df_feat_values['Feature'].values,
                y = y1,
                mode = 'lines+markers',
                line=dict(color='lightgreen', width=2),
                name = 'mean of positive loans',
    ))
    fig_feat.add_trace(go.Scatter(
                x =  df_feat_values['Feature'].values,
                y =  y2,
                mode = 'lines+markers',
                line=dict(color='black', width=2),
                name = 'value of current loan',
    ))
    fig_feat.add_trace(go.Scatter(
                x =  df_feat_values['Feature'].values,
                y =  y3,
                mode = 'lines+markers',
                line=dict(color='lightsalmon', width=2),
                name = 'mean of negative loans',
    ))
    fig_feat.update_layout(plot_bgcolor='white')

    # insert score in the good range of false positive rates (risks)
    new_risk_marks = calc_slider_marks(risk_fpr_list, '100%', '0%')
    risk_thr, left_risk_mark, right_risk_mark, new_risk_marks\
        = update_slider_marks(score, new_risk_marks, risk_fpr_list)

    # update gauge with score and false positive ranges (risk)
    fig_risk = go.Figure(go.Indicator(
                domain={'x': [0, 1], 'y': [0, 1]},
                value=score,
                mode='gauge+number',
                title={'text': label_gauge_risk,
                       'font': {'size': font_size_gauge_title}},
                gauge={
                    'axis': {'range': [None, 1.0]},
                    'steps':steps_gauge_risk,   
                    'threshold': {'line': {'color': 'green', 'width': 2},
                                  'thickness': 0.75, 'value': score}
                }
            ))

    # insert score in good range of false neg rates (loss of earnings)
    new_turnover_marks = calc_slider_marks(turnover_fnr_list, '0%', '100%')
    turnover_thr, left_turnover_mark, right_turnover_mark, new_turnover_marks\
        = update_slider_marks(score, new_turnover_marks, turnover_fnr_list)

    # update gauge with score and false neg ranges (loss of earnings)
    fig_turnover = go.Figure(go.Indicator(
                domain={'x': [0, 1], 'y': [0, 1]},
                value=score,
                mode='gauge+number',
                title={'text': label_gauge_turnover,
                       'font': {'size': font_size_gauge_title}},
                gauge={'axis': {'range': [None, 1.0]},
                    'steps': steps_gauge_turnover,
                    'threshold': {'line': {'color': 'green', 'width': 2},
                                  'thickness': 0.75, 'value': score}
                }
            ))

    # recommendation of Credit Home according to score
    reco_label, reco_image  = get_recommendation(score)

    return\
        fig_feat,\
        lb_feat_values,\
        df_feat_values.to_dict('records'),\
        f'{label_risk_beg}{risk_thr}{label_risk_end}',\
        fig_risk,\
        new_risk_marks,\
        [left_risk_mark, right_risk_mark],\
        f'{label_turnover_beg}{turnover_thr}{label_turnover_end}',\
        fig_turnover,\
        new_turnover_marks,\
        [left_turnover_mark, right_turnover_mark],\
        app.get_asset_url(reco_image),\
        reco_label


if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=8050, debug=False)