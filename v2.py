from datetime import timedelta

import streamlit as st

from chart_gpt.assistant import GlobalResources
from chart_gpt.assistant import StateActions
from chart_gpt.utils import AssistantFrame
from chart_gpt.utils import UserFrame


@st.cache_resource(ttl=timedelta(hours=24))
def global_resources() -> GlobalResources:
    return GlobalResources.initialize()


if 'state_actions' not in st.session_state:
    st.session_state.state_actions = StateActions(resources=global_resources())

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
    assistant_message = st.chat_message("assistant")
    try:
        with st.spinner("Writing query"):
            assistant_frame.query = st.session_state.state_actions.generate_query()
            assistant_frame.render(assistant_message)
        with st.spinner("Running query"):
            assistant_frame.result_set = st.session_state.state_actions.run_query()
            assistant_frame.render(assistant_message)
        with st.spinner("Summarizing data"):
            assistant_frame.summary = st.session_state.state_actions.summarize_data()
            assistant_frame.render(assistant_message)
        with st.spinner("Rendering chart"):
            assistant_frame.chart = st.session_state.state_actions.visualize_data()
            assistant_frame.render(assistant_message)
    except (Exception,) as e:
        assistant_frame.error = str(e)
        assistant_frame.render(assistant_message)
