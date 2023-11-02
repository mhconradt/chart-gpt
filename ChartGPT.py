import json
import logging

from datetime import timedelta

import streamlit as st

from chart_gpt.assistant import GenerateQueryCommand
from chart_gpt.assistant import GenerateQueryOutput

from chart_gpt.assistant import GlobalResources
from chart_gpt.assistant import Interpreter
from chart_gpt.assistant import RunQueryCommand
from chart_gpt.assistant import RunQueryOutput
from chart_gpt.assistant import SessionState
from chart_gpt.assistant import StateActions
from chart_gpt.assistant import SummarizeResultSetCommand
from chart_gpt.assistant import SummarizeResultSetOutput
from chart_gpt.assistant import VisualizeResultSet
from chart_gpt.assistant import VisualizeResultSetOutput
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

    You can try it out using the TPCDS dataset or configure your own database using the 
    "🔒 Credentials" tab. Currently the only supported database is Snowflake, but we're working on
    adding support for more databases like Redshift, BigQuery, Postgres, and MySQL.

    If you're not sure what to ask, you can start with "Show me tables in the database." 

    Enjoy! 
""")

st.title("ChartGPT 📈")

if 'secrets' not in st.session_state:
    st.session_state.secrets = {}

salt = hash(json.dumps(st.session_state.secrets))


class StreamlitStateActions(StateActions):
    def generate_query(self, command: GenerateQueryCommand) -> GenerateQueryOutput:
        with st.spinner("Writing query"):
            generate_query_output = super().generate_query(command)
            assistant_frame.query = generate_query_output.query
            assistant_frame.render(placeholder)
            return generate_query_output

    def run_query(self, command: RunQueryCommand) -> RunQueryOutput:
        with st.spinner("Running query"):
            run_query_output = super().run_query(command)
            result_set_id = run_query_output.result_set_id
            assistant_frame.result_set = self.state.result_sets[result_set_id]
            assistant_frame.render(placeholder)
            return run_query_output

    def summarize_result_set(self, command: SummarizeResultSetCommand) -> SummarizeResultSetOutput:
        with st.spinner("Gathering insights"):
            summarize_result_set_output = super().summarize_result_set(command)
            assistant_frame.summary = summarize_result_set_output.summary
            assistant_frame.render(placeholder)
            return summarize_result_set_output

    def visualize_result_set(self, command: VisualizeResultSet) -> VisualizeResultSetOutput:
        with st.spinner("Crafting visualization"):
            visualize_result_set_output = super().visualize_result_set(command)
            assistant_frame.chart = visualize_result_set_output.vega_lite_specification
            assistant_frame.render(placeholder)
            return visualize_result_set_output


if 'session_state' not in st.session_state:
    st.session_state.session_state = SessionState()


@st.cache_resource(ttl=timedelta(hours=4), show_spinner=False)
def get_state_actions(salt):
    with st.spinner("Initializing"):
        try:
            secrets = {**st.secrets, **st.session_state.secrets}
            global_resources = GlobalResources.initialize(secrets=secrets)
            actions = StateActions(resources=global_resources.model_dump(),
                                   session_state=st.session_state.session_state)
            st.success(f"Initialized schema {global_resources.connection.schema} in database {global_resources.connection.database}")
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
    st.session_state.state_actions.add_message(dict(content=prompt, role="user"))
    interpreter = Interpreter(actions=st.session_state.state_actions)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            assistant_frame.final = interpreter.run()
            assistant_frame.render(placeholder)
        except (Exception,) as e:
            assistant_frame.error = str(e)
            assistant_frame.render(placeholder)
            logger.exception("Error while answering question %s", e)
