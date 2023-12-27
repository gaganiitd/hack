import streamlit as st
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import json

with st.form('form'):
    stock_name = st.selectbox('stock name',['TCS.NS','HDFCBANK.NS', 'RELIANCE.NS','INFY.NS','ICICIBANK.NS','HINDUNILVR.NS','AXISBANK.NS','BHARTIARTL.NS','MARUTI.NS','LT.NS'])
    year = st.number_input("Year", value=2022, max_value=2023, min_value=2018)
    pred = st.selectbox("Predict", [1, 2, 3, 4, 5, 6])
    submit = st.form_submit_button('Submit')


if submit:
    df = pd.read_csv("./csv/{}.csv".format(stock_name))
    df = pd.DataFrame(df)[["date", "close"]]
    df["date"] = pd.to_datetime(df["date"])

    include = df[df["date"].dt.year == year]

    st.line_chart(include, x="date", y="close")
    preds_path = f"./csv/{stock_name}_preds.csv"
    if preds_path:
        df_pred = pd.read_csv("./csv/{}_preds.csv".format(stock_name))
        df_pred = pd.DataFrame(df_pred)[["date", "close"]]
        datetime_str = dt.datetime.now()

        today = datetime_str
        future_date_after_2days = today + \
                                 timedelta(days = 30*pred)
        df_pred["date"] = pd.to_datetime(df_pred["date"])

        df2 = df_pred[(df_pred['date'] > str(today)) & (df_pred['date']<str(future_date_after_2days))]


        st.line_chart(df2, x="date", y="close")