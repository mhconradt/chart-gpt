import os
from typing import Any
from typing import Literal
from typing import Mapping

import openai
from pandas import DataFrame
from pydantic import Field
from snowflake.connector import DictCursor
from snowflake.connector import NotSupportedError
from snowflake.connector import SnowflakeConnection

from chart_gpt import ChartIndex
from chart_gpt import DatabaseCrawler
from chart_gpt import get_connection
from chart_gpt.charts import ChartGenerator
from chart_gpt.sql import SQLGenerator
from chart_gpt import chat_summarize_data
from chart_gpt.utils import ChartGptModel

EmptyListField = Field(default_factory=list)


class SessionState(ChartGptModel):
    # A materialized view of all chat messages
    messages: list[dict] = EmptyListField
    queries: list[str] = EmptyListField
    result_sets: list[DataFrame] = EmptyListField
    execution_errors: list[str] = EmptyListField
    summaries: list[str] = EmptyListField
    charts: list[dict] = EmptyListField

    @property
    def last_user_message(self) -> str:
        for message in reversed(self.messages):
            if message["role"] == "user":
                return message["content"]
        else:
            raise LookupError("user")


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


class StateActions(ChartGptModel):
    state: SessionState = Field(default_factory=SessionState)
    resources: GlobalResources

    def add_message(self, content: str, role: Literal["user", "assistant"]):
        self.state.messages.append({"role": role, "content": content})

    def generate_query(self) -> str:
        try:
            query = self.resources.sql_generator.generate_valid_query(self.state.last_user_message)
            self.state.queries.append(query)
            return query
        except LookupError:
            raise UnsupportedAction()

    def run_query(self) -> DataFrame:
        try:
            last_query = self.state.queries[-1]
        except IndexError:
            raise UnsupportedAction()
        try:
            cursor = self.resources.connection.cursor(cursor_class=DictCursor).execute(last_query)
            try:
                result_set = cursor.fetch_pandas_all()
            except NotSupportedError:
                result_set = DataFrame(cursor.fetchall())
            self.state.result_sets.append(result_set)
            return result_set
        except (Exception,) as e:
            # Not sure about this.
            self.state.execution_errors.append(e.args[0])
            raise e

    def summarize_data(self) -> str:
        try:
            summary = chat_summarize_data(
                result_set=self.state.result_sets[-1],
                question=self.state.last_user_message,
                query=self.state.queries[-1]
            )
            self.state.summaries.append(summary)
            return summary
        except IndexError:
            raise UnsupportedAction()

    def visualize_data(self) -> dict:
        try:
            chart = self.resources.chart_generator.generate(
                question=self.state.last_user_message,
                query=self.state.queries[-1],
                result=self.state.result_sets[-1],
            )
            self.state.charts.append(chart)
            return chart
        except (LookupError, IndexError):
            raise UnsupportedAction()
