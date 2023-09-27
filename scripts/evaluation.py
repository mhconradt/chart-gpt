import pandas as pd
from pandas import DataFrame

from chart_gpt import DatabaseCrawler
from chart_gpt import SQLIndex
from chart_gpt import get_connection


def precision_recall_f1(a, b):
    a, b = set(a), set(b)
    intersection = (a & b)
    try:
        precision = len(intersection) / len(a)
    except ZeroDivisionError:
        precision = None

    try:
        recall = len(intersection) / len(b)
    except ZeroDivisionError:
        recall = None
    f1 = None
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except (TypeError, ZeroDivisionError):
        f1 = None
    return precision, recall, f1


def main():
    conn = get_connection()
    crawler = DatabaseCrawler(conn)
    index_data = crawler.get_index_data()
    index = SQLIndex.from_data(index_data)
    reference_queries = pd.read_json('data/TPC-DS-tables-columns.json')
    index_top_tables = reference_queries['question'].map(index.top_tables)
    index_top_columns = reference_queries['question'].map(
        lambda question: [c for t, c in index.top_columns(question)]
    )
    table_precision_recall = DataFrame([
        precision_recall_f1(actual, test)
        for actual, test in
        zip(reference_queries['tables'], index_top_tables)
    ], columns=['precision', 'recall', 'f1'])
    column_precision_recall = DataFrame([
        precision_recall_f1(actual, test)
        for actual, test in
        zip(reference_queries['columns'], index_top_columns)
    ], columns=['precision', 'recall', 'f1'])
    print(pd.concat([table_precision_recall, column_precision_recall], keys=['table', 'column'], axis=1).describe())
    pass


if __name__ == '__main__':
    main()
