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
import pandas as pd
from pages import Pages


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

@st.cache(ttl=600)
def worksheet_names():
    sheet_names = []
    for sheet in worksheet_list:
        sheet_names.append(sheet.title)
    return sheet_names

def load_spreadsheet(spreadsheetname):
    worksheet = sh.worksheet(spreadsheetname)
    df = pd.DataFrame(worksheet.get_all_records())
    return df

def update_spreadsheet(spreadsheetname, df):
    spread.df_to_sheet(df, sheet=spreadsheetname, index=False)
    st.info("Atualizado na Planilha !!!")


lista_menu = ['Controle de Posição', 'Mercado','Modelos', 'Relatórios']
st.sidebar.subheader('Menu Principal')
escolha = st.sidebar.radio('Escolha a Opção: ', lista_menu)

# Escolha das páginas
pg = Pages()

if escolha == 'Controle de Posição':
    df = load_spreadsheet('positions_BrunoBariotto')
    pg.controle(df)
    
if escolha == 'Mercado':
    df = load_spreadsheet('positions_BrunoBariotto')
    pg.mercado(df)

if escolha == 'Modelos':
    pg.modelos()
    
if escolha == 'Relatórios':
    pg.relatorios()


#st.header('Controle de Posição - Bruno Bariotto')



#what_sheets = worksheet_names()

#ws_choice = st.sidebar.radio('Escolha a Aba desejada', what_sheets)

#st.write(ws_choice)
#df = load_spreadsheet(ws_choice)
#st.write(df)

