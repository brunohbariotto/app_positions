# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 15:03:10 2023

@author: Dell inspiron
"""

from google.oauth2 import service_account
from gspread_pandas import Spread, Client
import streamlit as st
import pandas as pd

class PlanGoogle:
    def __init__(self):
        self.scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
        self.client = self.init_credentials()
        self.spreadsheetname = "Position_Control"
        
    #
    # inicializa as credenciais do google planilhas
    #    
    def init_credentials(self):
        
        credentials = service_account.Credentials.from_service_account_info(
             st.secrets["gcp_service_account"],
             scopes=self.scope)

        client = Client(scope=self.scope, creds=credentials)
        return client
    
    # lê e retorna um dataframe de uma aba (tabname) da planilha inicializada
    def read_spreadsheet(self, tabname):
        spread = Spread(self.spreadsheetname, client = self.client)
        st.write(spread.url)

        sh = self.client.open(self.spreadsheetname)
        worksheet = sh.worksheet(tabname)
        df = pd.DataFrame(worksheet.get_all_records())
        return df

    # @st.cache(ttl=600)
    # def worksheet_names():
    #     sheet_names = []
    #     for sheet in worksheet_list:
    #         sheet_names.append(sheet.title)
    #     return sheet_names

    # def load_spreadsheet(spreadsheetname):
    #     worksheet = sh.worksheet(spreadsheetname)
    #     df = pd.DataFrame(worksheet.get_all_records())
    #     return df

    # def update_spreadsheet(spreadsheetname, df):
    #     spread.df_to_sheet(df, sheet=spreadsheetname, index=False)
    #     st.info("Atualizado na Planilha !!!")