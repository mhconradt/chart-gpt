import glob
import json

import numpy as np
import openai
from openai.embeddings_utils import get_embedding
from pandas import DataFrame
from pandas import Series
from pydantic import BaseModel

from chart_gpt.utils import extract_json
from chart_gpt.utils import generate_completion
from chart_gpt.utils import json_dumps_default
from chart_gpt.utils import pd_vss_lookup

CHART_CONTEXT_EXCLUDE_EXAMPLES = [
    "layer_likert",  # 4532
    "isotype_bar_chart",  # 3924
    "interactive_dashboard_europe_pop",  # 3631
    "layer_line_window"  # 3520
]
CHART_DEFAULT_CONTEXT_ROW_LIMIT = 10
VEGA_LITE_CHART_PROMPT_FORMAT = """
Examples:
{examples}
Generate a Vega Lite chart that answers the user's question from the data.
User question: {question}
Generated SQL query: {query}
SQL query result (these will be automatically included in data.values):
{result}
Vega-Lite definition following schema at https://vega.github.io/schema/vega-lite/v5.json:
"""
CHART_EMBEDDING_MODEL = 'text-embedding-ada-002'


class ChartIndex(BaseModel):
    # [chart_id] -> [specification]
    specifications: DataFrame
    # [chart_id] -> [dim_0, dim_1, ..., dim_n]
    embeddings: DataFrame

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(cls) -> "ChartIndex":
        vl_example_files = glob.glob('data/vega-lite/examples/*.vl.json')
        vl_example_names = [
            example.removeprefix('data/vega-lite/examples/').removesuffix('.vl.json')
            for example in vl_example_files
        ]
        # vl_example_specs = [json.load(open(example, 'r')) for example in vl_example_files]
        vl_example_specs_json = [open(example, 'r').read() for example in vl_example_files]
        response = openai.Embedding.create(
            input=vl_example_specs_json,
            engine=CHART_EMBEDDING_MODEL
        )
        embeddings = [item.embedding for item in response.data]
        embeddings_df = DataFrame(
            embeddings,
            index=vl_example_names
        ).rename(lambda i: f"dim_{i}", axis=1).drop(CHART_CONTEXT_EXCLUDE_EXAMPLES)
        specifications_df = DataFrame(
            vl_example_specs_json,
            index=vl_example_names,
            columns=['specification']
        ).drop(CHART_CONTEXT_EXCLUDE_EXAMPLES)
        return ChartIndex(embeddings=embeddings_df, specifications=specifications_df)

    def top_charts(self, question: str, data: DataFrame) -> Series:
        """
        Finds the most relevant chart specifications to the given question and data.
        :return: Series [chart_id] -> specification
        """
        embedding_query_string = json.dumps(
            {
                "title": question,
                "data": {
                    "values": data.head(CHART_DEFAULT_CONTEXT_ROW_LIMIT).to_dict(orient="records")
                }
            },
            default=json_dumps_default
        )
        embedding_query = np.array(
            get_embedding(embedding_query_string, engine=CHART_EMBEDDING_MODEL)
        )
        chart_ids = pd_vss_lookup(self.embeddings, embedding_query, n=3).index.tolist()
        return self.specifications.specification.loc[chart_ids]


class ChartGenerator:
    def __init__(self, index: ChartIndex):
        self.index = index

    def generate(self, question: str, query: str, result: DataFrame) -> dict:
        data_values = result.to_dict(orient='records')
        prompt = VEGA_LITE_CHART_PROMPT_FORMAT.format(
            result=json.dumps(data_values[:CHART_DEFAULT_CONTEXT_ROW_LIMIT],
                              default=json_dumps_default),
            question=question,
            query=query,
            examples=json.dumps(
                self.index.top_charts(question, result).map(json.loads).tolist(),
                default=json_dumps_default
            )
        )
        completion = generate_completion(prompt)
        specification = extract_json(completion)
        specification['data'] = {'values': data_values}
        return specification
