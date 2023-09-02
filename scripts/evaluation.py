import json

from pandas import DataFrame

from chart_gpt import DataBoss
from chart_gpt import Index
from chart_gpt import IndexData
from chart_gpt import get_connection


def main():
    with open('../data/TPCDS_SF10TCL-queries.json', 'r') as f:
        queries = json.load(f)
    # [query] -> [query]
    query_df = DataFrame(queries, index=queries, columns=['query']).rename_axis('query')
    print(query_df)
    conn = get_connection()
    boss = DataBoss(conn)
    n = 24
    index_data = IndexData(
        samples=boss.get_table_samples(n_tables=n, n_rows=0),
        descriptions=boss.get_table_descriptions(n=n)
    )
    index = Index.from_data(index_data)
    score_df = query_df.apply(lambda s: index.top_tables(s.query, n=n), axis=1)
    print(score_df)


if __name__ == '__main__':
    main()
