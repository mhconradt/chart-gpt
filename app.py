import traceback

import streamlit as st

from chart_gpt import chat_summarize_data
from chart_gpt import create_index
from chart_gpt import display_data
from chart_gpt import generate_valid_sql
from chart_gpt import get_connection

# Question

question = st.text_input("What questions do you have about your data?")


@st.cache_resource
def c_get_connection():
    return get_connection()


conn = c_get_connection()


@st.cache_resource
def c_create_index():
    return create_index(conn)


index = c_create_index()

query = generate_valid_sql(conn, index, question)

st.text(query)

# Data

df = conn.cursor().execute(query).fetch_pandas_all()

st.write(df)

try:
    viz = display_data(df, conn)

    st.altair_chart(viz)
except Exception as e:
    traceback.print_exc()

try:
    st.text(chat_summarize_data(df, question))
except Exception:
    traceback.print_exc()

# Chart
