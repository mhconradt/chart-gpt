import json
import logging

from datetime import timedelta

import streamlit as st

from chart_gpt.assistant import GlobalResources
from chart_gpt.assistant import StateActions
from chart_gpt.frame import AssistantFrame
from chart_gpt.frame import UserFrame

log_level = st.secrets.get("LOG_LEVEL", "DEBUG")

logging.getLogger("chart_gpt.sql").setLevel(log_level)
logging.getLogger("chart_gpt.charts").setLevel(log_level)
logging.getLogger("chart_gpt.utils").setLevel(log_level)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

st.set_page_config(
    page_title="ChartGPT",
    page_icon="📈"
)

st.markdown("""
    Welcome to ChartGPT 🎉 

    You can try it out using the TPCDS dataset or configure your own database using the Credentials 
    tab. 

    If you're not sure what to ask, you can start with "Show me tables in the database." 

    Enjoy! 
""")

st.title("ChartGPT 📈")

if 'secrets' not in st.session_state:
    st.session_state.secrets = {}

salt = hash(json.dumps(st.session_state.secrets))


@st.cache_resource(ttl=timedelta(hours=4), show_spinner=False)
def get_state_actions(salt):
    with st.spinner("Initializing"):
        try:
            secrets = {**st.secrets, **st.session_state.secrets}
            global_resources = GlobalResources.initialize(secrets=secrets)
            actions = StateActions(resources=global_resources.model_dump())
            st.success(f"Initialized database {global_resources.connection.database}")
            return actions
        except (Exception,) as e:
            st.error(e)


st.session_state.state_actions = get_state_actions(salt)

if 'frames' not in st.session_state:
    st.session_state.frames = []

for frame in st.session_state.frames:
    role = "user" if isinstance(frame, UserFrame) else "assistant"
    history_message = st.chat_message(role)
    frame.render(history_message)

if prompt := st.chat_input("What questions do you have about your data?"):
    user_frame = UserFrame(prompt=prompt)
    st.session_state.frames.append(user_frame)
    user_message = st.chat_message("user")
    user_frame.render(user_message)

    assistant_frame = AssistantFrame()
    st.session_state.frames.append(assistant_frame)
    st.session_state.state_actions.add_message(prompt, role="user")
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            with st.spinner("Writing query"):
                assistant_frame.query = st.session_state.state_actions.generate_query()
                assistant_frame.render(placeholder)
            with st.spinner("Running query"):
                assistant_frame.result_set = st.session_state.state_actions.run_query()
                assistant_frame.render(placeholder)
            if len(assistant_frame.result_set) > 1:
                with st.spinner("Summarizing data"):
                    assistant_frame.summary = st.session_state.state_actions.summarize_data()
                    assistant_frame.render(placeholder)
                with st.spinner("Rendering chart"):
                    assistant_frame.chart = st.session_state.state_actions.visualize_data()
                    assistant_frame.render(placeholder)
        except (Exception,) as e:
            assistant_frame.error = str(e)
            assistant_frame.render(placeholder)
            logger.exception("Error while answering question %s", e)
