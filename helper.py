import pandas as pd
import numpy as np
from datetime import date, timedelta
from tqdm import tqdm
import time
from dotenv import load_dotenv
import statsmodels.tsa.api as smt
import os
import ast
import requests

# --- Prelims

# --- Data


def get_data_from_fred(id: str, api: str):
    url = (
        "https://api.stlouisfed.org/fred/series/observations?series_id="
        + id
        + "&api_key="
        + api
        + "&file_type=json"
    )
    response = requests.get(url)
    df = pd.DataFrame(response.json()["observations"])[["date", "value"]]
    return df


def x13_deseasonalise(data: pd.DataFrame, cols_to_adj: list[str]):
    # deep copy
    df = data.copy()
    # adjust column by column
    for col in cols_to_adj:
        # run x13
        res = smt.x13_arima_analysis(endog=df[col])
        # extract the deseasonalised series
        df[col] = res.seasadj
    # output
    return df