from dash import Dash, dcc, html, Input, Output
# import dash_core_components as dcc
# import dash_html_components as html
import pandas as pd
from mysql import connector

db_connection = connector.connect(
  host="localhost",
  user="root",
  passwd="pratimbhagat2001",
  db="users"
)

cursor = db_connection.cursor()

df = pd.read_sql("select * from user;", db_connection,['username', 'email', 'password'])

def generate_table(dataframe):
   return html.Table(className='hellob',
      # Header
      children=[html.Tr([html.Th(col) for col in dataframe.columns])] +
      # Body
      [html.Tr([
         html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
      ]) for i in range(len(dataframe))]
   )
    
# app = dash.Dash()
# app.layout = html.Div(children=[
#    html.H4(children='Sales Of February 2020 by MHD'),

#    html.Label('Product Name: '),
#    dcc.Input(
#        id='tfield',
#        placeholder='Input the product name',
#        type='text'
#        ),
#    generate_table(df)
# ])




app = Dash()
app.layout = html.Div(children=[
   html.H4(children='Users Search'),
   html.Label('UserName: '),
   dcc.Input(
       id='tfield',
       placeholder='Input the product name',
       type='text'
       ),
   html.Br(),html.Br(),
   html.Div(id='my-output'),
])

@app.callback(Output('my-output','children'),[Input('tfield', 'value')])
def callback(input_value):
   df = pd.read_sql("select * from user ", db_connection,params= (input_value,))
   return generate_table(df)

if __name__ == '__main__':
   app.run_server(debug=False)