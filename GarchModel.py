import dash

from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas_datareader.data as web
from datetime import datetime as dt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from arch import arch_model


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div([     #INIZIO GRAFICA
                           
    html.Div([html.Strong("TARCH CALCOLATOR")], style={'font-family':'sans-serif', 'font-size':30, 'text-align': 'center'}),
    html.Hr(style={'border':'3px solid black'}),
    
    html.Br(),
    html.Br(),
    
    html.Div([
        html.Div([
            #TENDA NOME INDICE
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            dcc.DatePickerRange(
                id='date_picker_range',
                style={'font-family':'sans-serif', 'width':'100%', 'margin-left':'80px'},
                min_date_allowed = dt(1990, 1, 1),
                max_date_allowed = dt.today(),
                start_date = dt(2015,1,1),
                end_date = dt.today(),
                initial_visible_month = dt.today(),
                ),
            html.Br(),
            html.Br(),
            dcc.Dropdown(
                id='index_name',
                style={'font-family':'sans-serif', 'width':'70%', 'margin-left':'40px'},
                options=[
                    {'label': 'S&P500', 'value': '^GSPC'},
                    {'label': 'Nasdaq', 'value': '^IXIC'},
                    {'label': 'Nikkei 225', 'value': '^N225'},
                    {'label': 'HANG SENG', 'value': '^HSI'},
                    {'label': 'IPC MEXICO', 'value': '^MXX'},
                    {'label': 'CAC 40', 'value': '^FCHI'}
                    ], value='^GSPC'),
            html.Br(),
    
            #TENDA TIPO DI RITORNO
            dcc.Dropdown(
                id='index_return',
                style={'font-family':'sans-serif', 'width':'70%', 'margin-left':'40px'},
                options=[
                    {'label': 'Simple return', 'value': 'Adj Close'},
                    {'label': 'Log return', 'value': 'Log Return'},
                    {'label': 'Log return^2', 'value': 'Log Return2'}
                    ], value='Adj Close'),
            html.Br(),
    
            #TENDA GRANULARITA'
            dcc.Dropdown(
                id='granularity_type',
                style={'font-family':'sans-serif', 'width':'70%', 'margin-left':'40px'},
                options=[
                    {'label': 'Day', 'value': 'D'},
                    {'label': 'Week', 'value': 'W'},
                    {'label': 'Month', 'value': 'M'},
                    {'label': 'Year', 'value': 'Y'}
                    ], value='W'),
            ], style={'verticalAlign': 'top','display': 'inline-block', 'width': '35%'}),
        
        #GRAFICO VISTA INIZALE
        html.Div([
            dcc.Graph(id='first_graph', style={'position':'static'}),
        ], style={'display': 'inline-block', 'width': '65%'}),
    ]),
    
    html.Br(),
    
    #SELETTORE TIPO MODELLO GARCH
    dcc.Dropdown(
        id='model_type',
        style={'font-family':'sans-serif', 'width':'50%', 'margin-left':'40px'},
        options=[
            {'label': 'TARCH', 'value': 'garch'},
            {'label': 'GJR-TARCH', 'value': 'gjr-garch'},
            ], value='garch'),
    
    html.Br(),
    html.Br(),
    
    #GRAAANDE CONTENITORE
    html.Div([
        html.Div([
            
            #TITOLO MODEL SUMMARY
            html.Div('Model summary:',
                     style={'font-family':'sans-serif', 'margin-left':'80px'}),
            #OUTPUT MODEL SUMMARY
            html.Pre(id='params',
                     style={'margin-left':'80px'}),
            
            ],style={'display':'inline-block'}),
        
        html.Div([
            
            #TITOLO
            html.Div('Ljung-Box test on differents lags, H0: no serial correlation',
                     style={'font-family':'sans-serif','margin-left':'40px'}),
            html.Br(),
            
            #PRIMO CONTENITORE
            html.Div([
                html.Div('Std Residuals',
                         style={'font-family':'sans-serif'}),
                html.Pre(id = 'ljung-box-std-res'),
                    ], style={'font-family':'sans-serif','display':'inline-block','margin-left':'40px'}),
            
            #SECONDO CONTENITORE
            html.Div([
                html.Div('Std Residuals^2',
                         style={'font-family':'sans-serif'}),
                html.Pre(id = 'ljung-box-std-res-sq'),
                    ], style={'font-family':'sans-serif','display':'inline-block','margin-left':'40px'}),
            
            #TERZO CONTENITORE
            html.Div([
                html.Div('Abs Std Residuals',
                         style={'font-family':'sans-serif'}),
                html.Pre(id = 'ljung-box-std-res-abs'),
                    ], style={'font-family':'sans-serif','display':'inline-block','margin-left':'40px'})
            
            ],style={'display':'inline-block','verticalAlign': 'top'})
        ]),
    
    #GRAFICI ACF RESIDUI
    html.Div([
        html.Div([
            dcc.Graph(id='acf_resid', style={'position':'static'})
            ], style={'display':'inline-block'}),
        html.Div([
            dcc.Graph(id='acf_resid_sq', style={'position':'static'})
            ], style={'display':'inline-block'}),
        html.Div([
            dcc.Graph(id='acf_resid_abs', style={'position':'static'})
            ], style={'display':'inline-block'}),
        ]),
    
    
    #GRAFICO COND VOLATILITY
    dcc.Graph(id='garch_output', style={'position':'static'}),
    
    
    #SEZIONE FORECAST
    html.Br(),
    html.Hr(),
    html.Div([html.Strong("Forecaster")], style={'font-family':'sans-serif', 'font-size':25, 'text-align': 'center'}),
    
    html.Br(),
    html.Div('Nota bene: il modello utilizzato per il forecast è lo stesso generato nella sezione sopra.', 
             style={'font-family':'sans-serif', 'margin-left':'40px'}),
    html.Br(),
    
    #SELETTORE STEP AHEAD
    dcc.Slider(1, 21, 1,
               value=1,
               id='step_ahead'),
    
    html.Br(),
    dcc.Graph(id='forecast_output', style={'position':'static'}),
   

                 
                      
]) #FINE GRAFICA

#OUTPUT ACTIONS
@app.callback(Output('first_graph', 'figure'), 				
              [Input('index_name', 'value'),
               Input('index_return', 'value'),
               Input('granularity_type', 'value'),
               Input('date_picker_range', 'start_date'),
               Input('date_picker_range', 'end_date')])

def update_graph(selected_dropdown_value,				
                 selected_dropdown_return,				
                 selected_dropdown_granularity,
                 start,
                 end):
    
    df = web.DataReader(selected_dropdown_value, data_source='yahoo', start=start, end=end)   
    df = df['Adj Close']
    df_ret = np.log(df/df.shift(1)) 
    df_ret = df_ret.dropna()
    df_ret_sq = df_ret**2
    df = df.to_frame()
    df_ret = df_ret.to_frame()
    df_ret_sq = df_ret_sq.to_frame()
    df_ret.columns = ["Log Return"]
    df_ret_sq.columns = ['Log Return2']
    df = df.join(df_ret)
    df = df.join(df_ret_sq)
    

    if(selected_dropdown_granularity=="M"):
        df = df.groupby(pd.Grouper(freq='M')).mean() 
    if(selected_dropdown_granularity=="W"):
        df = df.groupby(pd.Grouper(freq='W')).mean()
    if(selected_dropdown_granularity=='Y'):
        df = df.groupby(pd.Grouper(freq='Y')).mean()
    
    name = ""
    if(selected_dropdown_return=="Adj Close"):
        name = name + "Simple Return"
    else:
        name = name + "Continuous Compound Return"
    
    
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df[selected_dropdown_return].index, y=df[selected_dropdown_return], name=selected_dropdown_value, mode='lines'
        ))
                       
    fig.update_layout(title='Index graph', height=500, width=850)
    
    return fig   


@app.callback([Output('garch_output', 'figure'),
               Output('params','children'),
               Output('acf_resid','figure'),
               Output('acf_resid_sq','figure'),
               Output('acf_resid_abs','figure'),
               Output('ljung-box-std-res','children'),
               Output('ljung-box-std-res-sq','children'),
               Output('ljung-box-std-res-abs','children'),
               Output('forecast_output','figure')], 				
              [Input('index_name', 'value'),
               Input('granularity_type', 'value'),
               Input('model_type','value'),
               Input('date_picker_range', 'start_date'),
               Input('date_picker_range', 'end_date')],
               Input('step_ahead','value'))

def garch_model(selected_dropdown_value,			
                 selected_dropdown_granularity,
                 selected_dropdown_type,
                 start,
                 end,
                 step_ahead):
    
    df = web.DataReader(selected_dropdown_value, data_source='yahoo', start=start, end=end)  
    df = df['Adj Close']

    if(selected_dropdown_granularity=="M"):
        df = df.groupby(pd.Grouper(freq='M')).mean() 
    if(selected_dropdown_granularity=="W"):
        df = df.groupby(pd.Grouper(freq='W')).mean()
    if(selected_dropdown_granularity=='Y'):
        df = df.groupby(pd.Grouper(freq='Y')).mean()
    
    #df_ret = np.log(df/df.shift(1)) 
    #df_ret = df_ret.dropna()
    
    #Altri df ret
    df_ret = 100 * df.pct_change().dropna()
    
    if(selected_dropdown_type=='garch'):
        #garch = arch_model(df_ret, vol='garch', p=1, o=0, q=1)
        garch = arch_model(df_ret, vol='GARCH', power=2.0, p=1, o=0, q=1)
    elif(selected_dropdown_type=='gjr-garch'):
        #garch = arch_model(df_ret, vol='garch', p=1, o=1, q=1)
        garch = arch_model(df_ret, vol='GARCH', power=2.0, p=1, o=1, q=1)
    
    #MODELLO GARCH FITTATO
    garch_fitted = garch.fit(update_freq=5)
    
    cond_vol = garch_fitted.conditional_volatility.to_frame()
    
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=cond_vol.index, y=cond_vol['cond_vol']**2, name='Conditional Volatility', mode='lines'
        ))
    fig.add_trace(
        go.Scatter(x=df_ret.index, y=df_ret**2, name='Log Returns^2', mode='lines', opacity=0.5
        ))
    
    #OUTPUT PER CONDITIONAL VARIANCE                  
    fig.update_layout(title='Conditional Volatility', height=700)
    
    #OUTPUT PER PARAMETRI
    lista = []
    for phrase in str(garch_fitted).split('\n'):
        lista.append(phrase)
        lista.append(html.Br())
        
    #OUTPUT PER ACF STD RESID
    std_resid = garch_fitted.std_resid
    acf_std_resid = acf(std_resid)
    acf_std_resid = pd.DataFrame(acf_std_resid, columns=['acf'])
    fig_acf = px.bar(acf_std_resid[1:len(acf_std_resid)], x = [i for i in range(1,len(acf_std_resid))], y='acf')
    fig_acf.update_layout(title='Acf standardized residuals', height=500, width=450)
    
    #OUTPUT PER ACF STD RESID^2
    std_resid_sq = garch_fitted.std_resid**2
    acf_std_resid_sq = acf(std_resid_sq)
    acf_std_resid_sq = pd.DataFrame(acf_std_resid_sq, columns=['acf'])
    fig_acf_sq = px.bar(acf_std_resid_sq[1:len(acf_std_resid)], x = [i for i in range(1,len(acf_std_resid_sq))], y='acf')
    fig_acf_sq.update_layout(title='Acf standardized residuals^2', height=500, width=450)
    
    #OUTPUT PER ACF abs(STD RESID)
    std_resid_abs = abs(garch_fitted.std_resid)
    acf_std_resid_abs = acf(std_resid_abs)
    acf_std_resid_abs = pd.DataFrame(acf_std_resid_abs, columns=['acf'])
    fig_acf_abs = px.bar(acf_std_resid_abs[1:len(acf_std_resid)], x = [i for i in range(1,len(acf_std_resid_abs))], y='acf')
    fig_acf_abs.update_layout(title='Acf |standardized residuals|', height=500, width=450)

    #OUTPUT PER LJUNG-BOX TEST SUI STD RESID
    test_std_res = acorr_ljungbox(std_resid, lags=[1,2,3,4,5], return_df=True)
    test_std_res.columns = ['Test','p-value']
    test_std_res_sq = acorr_ljungbox(std_resid**2, lags=[1,2,3,4,5], return_df=True)
    test_std_res_sq.columns = ['Test','p-value']
    test_std_res_abs = acorr_ljungbox(abs(std_resid), lags=[1,2,3,4,5], return_df=True)
    test_std_res_abs.columns = ['Test','p-value']
    
    lista2 = []
    for phrase in str(test_std_res).split('\n'):
        lista2.append(phrase)
        lista2.append(html.Br())
        
    lista3 = []
    for phrase in str(test_std_res_sq).split('\n'):
        lista3.append(phrase)
        lista3.append(html.Br())
        
    lista4 = []
    for phrase in str(test_std_res_abs).split('\n'):
        lista4.append(phrase)
        lista4.append(html.Br())
    
    
    #FORECAST VOLATILITY
    forecast = garch_fitted.forecast(horizon=step_ahead, reindex=False, method="simulation")
    
    past = (cond_vol.values)**2
    
    pred = np.array([[fore] for fore in forecast.variance.values[0]])
    past_pred = np.append(past, pred)
    
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(x=np.arange(1,len(past_pred),1), y=past_pred, name='Forecast Volatility', mode='lines'
        ))
    #OUTPUT PER FORECAST               
    fig2.update_layout(title='Forecast Volatility', height=700)
    
    
    
    

    
    
    
    
    
    return fig, html.P(lista), fig_acf, fig_acf_sq, fig_acf_abs, html.P(lista2), html.P(lista3), html.P(lista4), fig2


if __name__ == '__main__':
    app.run_server()