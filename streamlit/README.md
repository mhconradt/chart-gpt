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

5. Create an OpenAI account and get an API key. This app costs very little
even with heavy usage (my highest bill was $20 for a whole month).

6. Set the necessary environment variables:
```bash
OPENAI_API_KEY=sk-***
SF_ACCOUNT=XYZ12345
SF_DATABASE=SNOWFLAKE_SAMPLE_DATA
SF_PASSWORD=$ecret
SF_ROLE=ACCOUNTADMIN
SF_SCHEMA=TPCH_SF1
```

7. Run the app:
```bash
streamlit run ChartGPT.py
```