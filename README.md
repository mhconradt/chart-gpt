## Installation

1. Install [Poetry](https://python-poetry.org/docs/#installation)

2. Set up a new virtual environment and install the requirements:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the requirements:

```bash
poetry install
```

4. Create a [Snowflake trial account](https://signup.snowflake.com/).
Free for 30 days and when you're done you create a new one.

5. Set the necessary environment variables:
```bash
OPENAI_API_KEY=sk-***
SF_ACCOUNT=XYZ12345
SF_DATABASE=SNOWFLAKE_SAMPLE_DATA
SF_PASSWORD=$ecret
SF_ROLE=ACCOUNTADMIN
SF_SCHEMA=TPCH_SF1
```

6. Run the app:
```bash
streamlit run ChartGPT.py
```