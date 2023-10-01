import json
from datetime import date
from datetime import datetime
from os import getenv

import numpy as np
import openai
from pandas import DataFrame
from pandas import Series
from snowflake.connector import connect


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


def generate_completion(system_prompt, model: str = 'gpt-4'):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt}
        ]
    ).choices[0].message.content
    return response


def extract_json(text: str, start: str = '{', stop: str = '}') -> dict:
    raw = text[text.index(start):text.rindex(stop) + 1]
    return json.loads(raw)


def json_dumps_default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(f"Object {o} not JSON serializable")


def pd_vss_lookup(index: DataFrame, query: np.array, n: int) -> Series:  # ?
    # Couldn't help myself but vectorize this
    index_norm = np.sqrt((index ** 2).sum(axis=1))
    query_norm = (query @ query) ** 0.5
    similarity = index @ query / (index_norm * query_norm)
    table_scores = similarity.sort_values(ascending=False) \
        .head(n)
    return table_scores
