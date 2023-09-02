from chart_gpt import e2e_qa
from chart_gpt import get_connection
from chart_gpt import create_index


def main():
    conn = get_connection()
    index = create_index(conn)
    question1 = "Show me the customers with the highest credit rating."
    e2e_qa(conn, index, question1)
    question2 = "Show me how many orders we've had in the last three months."
    e2e_qa(conn, index, question2)
    # Get tables and descriptions for query


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
