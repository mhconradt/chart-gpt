import inspect
import logging
import os
from typing import Any
from typing import Mapping
from uuid import uuid4

import openai
from pandas import DataFrame
from pydantic import Field
from snowflake.connector import DictCursor
from snowflake.connector import NotSupportedError
from snowflake.connector import SnowflakeConnection

from chart_gpt import ChartIndex
from chart_gpt import DatabaseCrawler
from chart_gpt import chat_summarize_data
from chart_gpt import get_connection
from chart_gpt.charts import ChartGenerator
from chart_gpt.schemas import ChartGptModel
from chart_gpt.sql import SQLGenerator

EmptyListField = Field(default_factory=list)

logger = logging.getLogger(__name__)


class SessionState(ChartGptModel):
    # A materialized view of all chat messages
    messages: list[dict] = EmptyListField
    result_sets: dict[str, DataFrame] = Field(default_factory=dict)


class GlobalResources(ChartGptModel):
    connection: SnowflakeConnection
    sql_generator: SQLGenerator
    chart_generator: ChartGenerator

    @classmethod
    def initialize(cls, secrets: Mapping[str, Any] = os.environ) -> "GlobalResources":
        connection = get_connection(secrets)
        openai.api_key = secrets.get('OPENAI_API_KEY')
        openai.organization = secrets.get('OPENAI_ORGANIZATION')
        crawler = DatabaseCrawler(connection=connection)
        index = crawler.get_index()
        sql_generator = SQLGenerator(connection=connection, index=index)
        chart_index = ChartIndex.create()
        chart_generator = ChartGenerator(index=chart_index)
        return GlobalResources(
            connection=connection,
            sql_generator=sql_generator,
            chart_generator=chart_generator
        )


class UnsupportedAction(Exception):
    pass


class GenerateQueryCommand(ChartGptModel):
    prompt: str = Field(description="A question / command containing any user hints or preferences on how to write the query. The LLM will do the rest.")


class GenerateQueryOutput(ChartGptModel):
    query: str = Field(description="A valid SQL query to run against the database.")


class RunQueryCommand(ChartGptModel):
    query: str = Field(description="A valid SQL query to run against the database.")


class RunQueryOutput(ChartGptModel):
    result_set_id: str = Field(description="A UUID that can be used in summarize_result_set and visualize_result_set.")
    columns: list[str] = Field(description="Columns present in the result set.")
    row_count: int = Field(description="Number of rows in the result set.")


class SummarizeResultSetCommand(ChartGptModel):
    prompt: str = Field(description="A question / command to answer / follow from the result set.")
    result_set_id: str = Field(description="A result set ID previously returned from run_query.")


class SummarizeResultSetOutput(ChartGptModel):
    summary: str = Field(description="A summary of how the result set answers the question / command.")


class VisualizeResultSet(ChartGptModel):
    prompt: str = Field(description="A question to be answered visually, or a command to be followed when generating the chart.")
    result_set_id: str = Field(description="A result set ID previously returned from run_query.")


class VisualizeResultSetOutput(ChartGptModel):
    vega_lite_specification: dict = Field(description="A Vega Lite specification.")


def get_openai_function(f):
    command_type = inspect.get_annotations(f)['command']
    function_info = {
        'name': f.__name__,
        'description': inspect.getdoc(f),
        'parameters': command_type.model_json_schema()
    }
    return function_info


class StateActions(ChartGptModel):
    state: SessionState = Field(default_factory=SessionState)
    resources: GlobalResources

    def add_message(self, message: dict):
        self.state.messages.append(message)

    def generate_query(self, command: GenerateQueryCommand) -> GenerateQueryOutput:
        """
        Uses an LLM to write a SQL query to answer a question / follow command.
        The query may be shown directly to the user by an external system.
        """
        try:
            query = self.resources.sql_generator.generate_valid_query(command.prompt)
            return GenerateQueryOutput(query=query)
        except LookupError:
            raise UnsupportedAction()

    def run_query(self, command: RunQueryCommand) -> RunQueryOutput:
        """
        Runs a SQL query and stores the result set for question answering and visualization.
        A preview of the data will be shown directly to the user by an external system.
        """
        try:
            cursor = self.resources.connection.cursor(cursor_class=DictCursor)
            cursor.execute(command.query)
            try:
                result_set = cursor.fetch_pandas_all()
            except NotSupportedError:
                result_set = DataFrame(cursor.fetchall())
            result_set_id = str(uuid4())
            self.state.result_sets[result_set_id] = result_set
            return RunQueryOutput(result_set_id=result_set_id,
                                  columns=list(result_set.columns),
                                  row_count=len(result_set))
        except (Exception,) as e:
            raise e

    def summarize_result_set(self, command: SummarizeResultSetCommand) -> SummarizeResultSetOutput:
        """
        Uses an LLM to summarize information in the result set relevant to a prompt. The summary
        will be shown directly to the user by an external system.
        """
        try:
            summary = chat_summarize_data(result_set=self.state.result_sets[command.result_set_id],
                                          question=command.prompt)
            return SummarizeResultSetOutput(summary=summary)
        except KeyError:
            raise UnsupportedAction()

    def visualize_result_set(self, command: VisualizeResultSet) -> VisualizeResultSetOutput:
        """
        Uses an LLM to create a Vega Lite specification to help answer the question / follow a command.
        This visualization will be shown directly to the user by an external system.
        """
        try:
            chart = self.resources.chart_generator.generate(
                question=command.prompt,
                result_set=self.state.result_sets[command.result_set_id]
            )
            return VisualizeResultSetOutput(vega_lite_specification=chart)
        except (LookupError, IndexError):
            raise UnsupportedAction()


class StreamlitStateActions(StateActions):
    def generate_query(self, command: GenerateQueryCommand) -> GenerateQueryOutput:
        return super().generate_query(command)

    def run_query(self, command: RunQueryCommand) -> RunQueryOutput:
        return super().run_query(command)

    def summarize_result_set(self, command: SummarizeResultSetCommand) -> SummarizeResultSetOutput:
        return super().summarize_result_set(command)

    def visualize_result_set(self, command: VisualizeResultSet) -> VisualizeResultSetOutput:
        return super().visualize_result_set(command)


class Interpreter(ChartGptModel):
    actions: StateActions

    def run(self) -> str:
        functions = [self.actions.generate_query, self.actions.run_query,
                     self.actions.summarize_result_set, self.actions.visualize_result_set]
        function_lut = {
            f.__name__: f
            for f in functions
        }
        openai_functions = [get_openai_function(f) for f in functions]
        while True:
            response = openai.ChatCompletion.create(
                model='gpt-4',
                messages=self.actions.state.messages,
                functions=openai_functions,
                temperature=0.0,
            ).choices[0]
            self.actions.add_message(response.message)
            if response.finish_reason == "function_call":
                function_call = response.message.function_call
                fn = function_lut[function_call.name]
                command_type = inspect.get_annotations(fn)['command']
                parsed_args = command_type.parse_raw(function_call.arguments)
                out = fn(parsed_args)
                self.actions.add_message({
                    "role": "function",
                    "name": function_call.name,
                    "content": out.json()
                })
            else:
                return response.message.content


def main():
    global_resources = GlobalResources.initialize()
    i = 1
    state = SessionState(
        messages=[
            {
                "role": "system",
                "content": """
                You are an assistant that uses tools to query databases, summarize data, and visualize data on behalf of the user.
                Some tools that you interact with are large language models with access to the relevant context.
                When using these tools, you create a prompt that contains all information relative to the model's task.
                """
            }
        ]
    )
    state_actions = StateActions(state=state, resources=global_resources)
    interpreter = Interpreter(actions=state_actions)
    while prompt := input(f"In [{i}]: "):
        state_actions.add_message({"role": "user", "content": prompt})
        output = interpreter.run()
        print(f"Out [{i}]:", output)


if __name__ == '__main__':
    main()
