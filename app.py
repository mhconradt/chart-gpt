import traceback

import streamlit as st

from chart_gpt import DatabaseCrawler
from chart_gpt import Index
from chart_gpt import SQLGenerator
from chart_gpt import chat_summarize_data
from chart_gpt import get_connection

# Question

question = st.text_input("What questions do you have about your data?")


@st.cache_resource
def c_get_connection():
    return get_connection()


conn = c_get_connection()

query_salt = 1234


@st.cache_resource
def c_index(_connection) -> Index:
    return DatabaseCrawler(_connection).get_index()


index = c_index(conn)

generator = SQLGenerator(conn, index)


@st.cache_resource
def generate_query(q, salt):
    print("generate query")
    return generator.generate_valid_query(q)


@st.cache_data
def run_query(q1, q2):
    cursor = conn.cursor()
    cursor.execute("alter session set query_tag = %(question)s;", {'question': q1})
    return cursor.execute(q2).fetch_pandas_all()


if question:
    if st.button("Regenerate query"):
        query_salt += 1

    with st.spinner("Generating query..."):
        query = generate_query(question, query_salt)

    if st.toggle("Show query"):
        st.code(query, language="sql")

    if st.checkbox("Run query?"):
        df = run_query(question, query)
        st.dataframe(df)

        st.text(chat_summarize_data(df, question, query))

#    if st.checkbox("Visualize data?"):
#        ...

# Data

#
# st.write(df)
#
# if len(df):
#     try:
#         # viz = display_data(df, conn)
#         #
#         # st.altair_chart(viz)
#         print('Skipping chart generation for query: ', question)
#     except Exception as e:
#         traceback.print_exc()
#
#     try:
#         st.text(chat_summarize_data(df, question))
#     except Exception:
#         traceback.print_exc()
#
# # Chart
#
