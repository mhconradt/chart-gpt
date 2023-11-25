import json
import logging
import os
from datetime import date
from datetime import datetime
from decimal import Decimal
from typing import Any
from typing import Mapping

import numpy as np
import openai
import tiktoken
from pandas import DataFrame
from pandas import Series
from snowflake.connector import SnowflakeConnection
from snowflake.connector import connect

logger = logging.getLogger(__name__)


def get_connection(secrets: Mapping[str, Any] = os.environ) -> SnowflakeConnection:
    return connect(
        account=secrets.get('SF_ACCOUNT'),
        user=secrets.get('SF_USER'),
        password=secrets.get('SF_PASSWORD'),
        role=secrets.get('SF_ROLE'),
        database=secrets.get('SF_DATABASE'),
        schema=secrets.get('SF_SCHEMA'),
    )


def generate_completion(prompt: str, model: str = 'gpt-4', temperature: float = 1.) -> str:
    logger.debug("Getting %s completion for prompt %s", model, prompt)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt}
        ],
        temperature=temperature
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


def get_token_count(text: str, model: str = 'gpt-4') -> int:
    return len(tiktoken.encoding_for_model(model).encode(text))
