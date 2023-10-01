from typing import Iterable
from typing import Optional

import pandas as pd
from pandas import DataFrame

from chart_gpt import DatabaseCrawler
from chart_gpt import SQLIndex
from chart_gpt import get_connection


def precision_recall_f1(
        a: Iterable[str],
        b: Iterable[str]
) -> tuple[Optional[float], Optional[float], Optional[float]]:
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
        f1 = compute_f1(precision, recall)
    except (TypeError, ZeroDivisionError):
        f1 = None
    return precision, recall, f1


def compute_f1(precision, recall):
    return 2 * precision * recall / (precision + recall)


def compute_evaluation_metrics(reference: pd.Series, retrieved: pd.Series) -> DataFrame:
    n_distinct = {y for x in reference for y in x}
    n_items, n_retrieved = len(n_distinct), retrieved.map(len).mean()
    baseline_precision = n_retrieved / n_items
    baseline_recall = n_retrieved / n_items
    baseline_f1 = compute_f1(baseline_precision, baseline_recall)
    stats = DataFrame(
        [precision_recall_f1(actual, test) for actual, test in zip(reference, retrieved)],
        columns=['precision', 'recall', 'f1']
    ).describe()
    stats.loc['baseline'] = pd.Series({
                                          "precision": baseline_precision,
                                          "recall": baseline_recall,
                                          "f1": baseline_f1
                                      })
    return stats


class EvaluationHarness:
    def __init__(self, index: SQLIndex):
        self.index = index

    def run(self, reference_queries: DataFrame) -> DataFrame:
        retrieved_tables = reference_queries['question'].map(self.index.top_tables)
        table_metrics = compute_evaluation_metrics(reference_queries['tables'], retrieved_tables)
        return table_metrics


def main():
    conn = get_connection()
    crawler = DatabaseCrawler(conn)
    index_data = crawler.get_index_data()
    index = SQLIndex.from_data(index_data)
    harness = EvaluationHarness(index)
    tpc_ds_metrics = harness.run(pd.read_json('data/TPC-DS-tables-columns.json'))
    print(tpc_ds_metrics.to_markdown())
    print()
    tpc_ds_lite_metrics = harness.run(pd.read_json('data/TPCDS-lite.json'))
    print(tpc_ds_lite_metrics.to_markdown())
    pass


if __name__ == '__main__':
    main()
