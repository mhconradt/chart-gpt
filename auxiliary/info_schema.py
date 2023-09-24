from datetime import timedelta

import streamlit as st
from pandas import DataFrame
from snowflake.connector import DictCursor

from chart_gpt import get_connection


@st.cache_resource(ttl=timedelta(hours=1))
def get_database_connection():
    connection = get_connection()
    connection.cursor().execute("use schema information_schema;").fetchone()
    return connection


conn = get_database_connection()


@st.cache_resource
def show_views():
    return DataFrame(conn.cursor(cursor_class=DictCursor).execute("show views;").fetchall())


views = show_views()

st.dataframe(views)

for view_name in views['name']:
    df = conn.cursor().execute("select * from identifier(%(view)s);", {"view": view_name}).fetch_pandas_all()
    st.text(view_name)
    st.dataframe(df)
