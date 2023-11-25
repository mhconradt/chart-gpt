from abc import abstractmethod
from typing import Optional

import streamlit as st
from pandas import DataFrame
from streamlit.delta_generator import DeltaGenerator

from chart_gpt.schemas import ChartGptModel


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
                st.markdown(self.summary)
            if self.chart is not None:
                st.vega_lite_chart(self.result_set, self.chart)
            if self.error is not None:
                st.error(self.error)


class UserFrame(Frame):
    prompt: str

    def render(self, canvas: DeltaGenerator):
        canvas.text(self.prompt)
