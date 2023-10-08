import streamlit as st

if 'secrets' not in st.session_state:
    st.session_state.secrets = {}

st.title("Credentials")

with st.form('Credentials'):
    st.caption('ChartGPT does not persistently store your credentials')

    user = st.text_input(label='User')

    password = st.text_input(label='Password', type='password')

    role = st.text_input(label='Role')

    account = st.text_input(label='Account')

    database = st.text_input(label='Database')

    schema = st.text_input(label='Schema')

    warehouse = st.text_input(label='Warehouse')

    openai_api_key = st.text_input(label='OpenAI API Key', type='password', placeholder='sk-...')

    if st.form_submit_button():
        form_data = {
            "OPENAI_API_KEY": openai_api_key,
            "SF_USER": user,
            "SF_ROLE": role,
            "SF_PASSWORD": password,
            "SF_ACCOUNT": account,
            "SF_DATABASE": database,
            "SF_SCHEMA": schema,
            "SF_WAREHOUSE": warehouse,
        }
        st.session_state.secrets = {k: v for k, v in form_data.items() if v}

        st.success("Credentials saved")
