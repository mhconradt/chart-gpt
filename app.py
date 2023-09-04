import traceback
from datetime import timedelta

import streamlit as st

from chart_gpt import ChartGenerator
from chart_gpt import ChartIndex
from chart_gpt import DatabaseCrawler
from chart_gpt import SQLIndex
from chart_gpt import SQLGenerator
from chart_gpt import chat_summarize_data
from chart_gpt import get_connection

# Question
st.title("ChartGPT")

question = st.text_input("What questions do you have about your data?")


@st.cache_resource(ttl=timedelta(hours=1))
def c_get_connection():
    return get_connection()


conn = c_get_connection()

query_salt = 1234


@st.cache_resource
def c_database_index(_connection) -> SQLIndex:
    return DatabaseCrawler(_connection).get_index()


@st.cache_resource
def c_chart_index() -> ChartIndex:
    return ChartIndex.create()


db_index = c_database_index(conn)

chart_index = c_chart_index()

query_generator = SQLGenerator(conn, db_index)

chart_generator = ChartGenerator(chart_index)


@st.cache_resource(show_spinner=False)
def generate_query(q, salt):
    print("generate query")
    return query_generator.generate_valid_query(q)


@st.cache_data(show_spinner=False)
def run_query(q1, q2):
    cursor = conn.cursor()
    cursor.execute("alter session set query_tag = %(question)s;", {'question': q1})
    return cursor.execute(q2).fetch_pandas_all()


@st.cache_resource(show_spinner=False)
def generate_chart(q1, q2, _result):
    return chart_generator.generate(q1, q2, _result)


if question:
    if st.button("Regenerate query"):
        query_salt += 1

    with st.spinner("Generating query..."):
        query = generate_query(question, query_salt)

    if st.toggle("Show query"):
        st.code(query, language="sql")

    if st.checkbox("Run query?"):
        with st.spinner("Running query..."):
            result = run_query(question, query)
            st.dataframe(result, hide_index=True)

        st.text(chat_summarize_data(result, question, query))

        if len(result) and st.checkbox("Visualize result?"):
            try:
                vega_lite_specification = chart_generator.generate(question, query, result)
                st.vega_lite_chart(result, vega_lite_specification)
            except Exception as e:
                traceback.print_exc()
