# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 15:03:10 2023

@author: Dell inspiron
"""

from google.oauth2 import service_account
from gspread_pandas import Spread, Client
import streamlit as st
import pandas as pd
from gspread.exceptions import WorksheetNotFound
from oscillator_optimization import SHEET, merge_saved, saved_windows

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
        #st.write(spread.url)

        sh = self.client.open(self.spreadsheetname)
        worksheet = sh.worksheet(tabname)
        df = pd.DataFrame(worksheet.get_all_records())
        return df

    def read_oscillator_windows(self):
        try:
            return self.read_spreadsheet(SHEET)
        except WorksheetNotFound:
            return pd.DataFrame()

    def save_oscillator_windows(self, results):
        # Re-read immediately before upsert; never replace the position worksheets.
        _, errors = saved_windows(results)
        if errors:
            raise ValueError('; '.join(errors))
        merged = merge_saved(self.read_oscillator_windows(), results)
        spread = Spread(self.spreadsheetname, client=self.client)
        spread.df_to_sheet(merged.fillna(''), sheet=SHEET, index=False, replace=True)

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

    def update_spreadsheet(self, tabname, df):
        spread = Spread(self.spreadsheetname, client = self.client)
        st.write('Enviando o DataFrame: ')
        st.write(df)
        spread.df_to_sheet(df, sheet=tabname, index=False, replace=True)
        st.info("Atualizado na Planilha !!!")
