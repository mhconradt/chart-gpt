import json
import random

import tiktoken

from chart_gpt import DatabaseCrawler
from chart_gpt import SQLGenerator
from chart_gpt import get_connection


def main():
    conn = get_connection()
    crawler = DatabaseCrawler(conn)
    index = crawler.get_index()
    generator = SQLGenerator(conn, index)
    with open('../data/TPCDS_SF10TCL-queries.json', 'r') as f:
        queries = json.load(f)
    gpt4_encoding = tiktoken.encoding_for_model('gpt-4')
    for query in random.Random(x=42).choices(queries):
        context = generator.get_context(query)
        print("# of tokens ", gpt4_encoding.encode(context))
        # print(f"""
        # Query: {query}
        # Context: {context}
        # """)


if __name__ == '__main__':
    main()
