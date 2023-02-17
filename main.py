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
from gsheetsdb import connect


st.write("New App Position!")
st.write("Trying")



# Create a connection object.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
    ],
)
conn = connect(credentials=credentials)

# Perform SQL query on the Google Sheet.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows

sheet_url = st.secrets["private_gsheets_url"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')

# Print results.
for row in rows:
    st.write(f"{row.name} has a :{row.pet}:")