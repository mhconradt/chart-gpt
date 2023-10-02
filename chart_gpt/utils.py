import json
import logging
from _decimal import Decimal
from abc import abstractmethod
from datetime import date
from datetime import datetime
from os import getenv
from typing import Optional

import numpy as np
import openai
from pandas import DataFrame
from pandas import Series
from pydantic import BaseModel
from snowflake.connector import connect
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

logger = logging.getLogger(__name__)


class ChartGptModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class Frame(ChartGptModel):
    @abstractmethod
    def render(self, canvas: DeltaGenerator):
        pass


class AssistantFrame(Frame):
    query: Optional[str] = None
    result_set: Optional[DataFrame] = None
    summary: Optional[str] = None
    chart: Optional[dict] = None
    error: Optional[str] = None

    def render(self, canvas: DeltaGenerator):
        with canvas.container():
            if self.query is not None:
                st.code(self.query, language="sql")
            if self.result_set is not None:
                st.dataframe(self.result_set, hide_index=True)
            if self.summary is not None:
                st.text(self.summary)
            if self.chart is not None:
                st.vega_lite_chart(self.result_set, self.chart)
            if self.error is not None:
                st.error(self.error)


class UserFrame(Frame):
    prompt: str

    def render(self, canvas: DeltaGenerator):
        canvas.text(self.prompt)


def get_connection():
    return connect(
        account=getenv('SF_ACCOUNT'),
        user=getenv('SF_USER'),
        password=getenv('SF_PASSWORD'),
        role=getenv('SF_ROLE'),
        database=getenv('SF_DATABASE'),
        schema=getenv('SF_SCHEMA'),
        # application='ChartGPT 0.0.0'
    )


def generate_completion(prompt: str, model: str = 'gpt-4') -> str:
    logger.debug("Getting %s completion for prompt %s", model, prompt)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt}
        ]
    ).choices[0].message.content
    logger.debug("Completion %s", response)
    return response


def extract_json(text: str, start: str = '{', stop: str = '}') -> dict:
    raw = text[text.index(start):text.rindex(stop) + 1]
    return json.loads(raw)


def json_dumps_default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    elif isinstance(o, Decimal):
        return float(o)
    raise TypeError(f"Object {o} not JSON serializable")


def pd_vss_lookup(index: DataFrame, query: np.array, n: int) -> Series:  # ?
    # Couldn't help myself but vectorize this
    index_norm = np.sqrt((index ** 2).sum(axis=1))
    query_norm = (query @ query) ** 0.5
    similarity = index @ query / (index_norm * query_norm)
    table_scores = similarity.sort_values(ascending=False) \
        .head(n)
    return table_scores
