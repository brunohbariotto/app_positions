# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 19:25:17 2023

@author: Dell inspiron
"""

import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from google.oauth2 import service_account
#from gsheetsdb import connect
from gspread_pandas import Spread, Client

st.write("New App Position!")
st.write("Trying")



# # Create a connection object.
# credentials = service_account.Credentials.from_service_account_info(
#     st.secrets["gcp_service_account"],
#     scopes=[
#         "https://www.googleapis.com/auth/spreadsheets",
#     ],
# )
# conn = connect(credentials=credentials)

# Create a google autentication object.
scope=["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]

credentials = service_account.Credentials.from_service_account_info(
     st.secrets["gcp_service_account"],
     scopes=scope)

client = Client(scope=scope, creds=credentials)
spreadsheetname = "Position_Control"

spread = Spread(spreadsheetname, client = client)

st.write(spread.url)

sh = client.open(spreadsheetname)
worksheet_list = sh.worksheets()

@st.cache_data(ttl=600)
def worksheet_names():
    sheet_names = []
    for sheet in worksheet_list:
        sheet_names.append(sheet.title)
    return sheet_names

st.header('Controle de Posição - Bruno Bariotto')
what_sheets = worksheet_names()

ws_choice = st.sidebar.radio('Escolha a Aba desejada', what_sheets)

st.write(ws_choice)

