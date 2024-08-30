# %%
import pandas as pd
import numpy as np
from datetime import date, timedelta
from helper import get_data_from_fred
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
import ast

time_start = time.time()

# %%
# 0 --- Main settings
load_dotenv()
fred_api_key = os.getenv("FRED_API_KEY")
path_data = "./raw-data/"
tel_config = os.getenv("TEL_CONFIG")
t_start = date(1947, 1, 1)

# %%
# I --- Pull data from FRED
dict_fred_urate = {
    "united_states": "UNRATE",
    "germany": "LRHUTTTTDEM156S",
    "france": "LRHUTTTTFRM156S",
    "italy": "LRHUTTTTITM156S",
    "united_kingdom": "LRHUTTTTGBM156S",
    "japan": "LRHUTTTTJPM156S",
    "australia": "LRUNTTTTAUM156S",
    # "singapore": "",
    "korea": "LRUNTTTTKRM156S",
    # "hong_kong": "",
    "mexico": "LRUNTTTTMXM156S",
    "chile": "LRUNTTTTCLM156S",
    "brazil": "LRUNTTTTBRM156S",
    # "china": "",
    # "india": "",
    # "malaysia": "",
    # "thailand": "",
    # "indonesia": "",
    # "philippines": "",
}
dict_country_names = {
    "united_states": "USA",
    "germany": "DEU",
    "france": "FRA",
    "italy": "ITA",
    "united_kingdom": "GBR",
    "japan": "JPN",
    "australia": "AUS",
    # "singapore": "SGP",
    "korea": "KOR",
    # "hong_kong": "HKG",
    "mexico": "MEX",
    "chile": "CHL",
    "brazil": "BRA",
    # "china": "CHN",
    # "india": "IND",
    # "malaysia": "MYS",
    # "thailand": "THA",
    # "indonesia": "IDN",
    # "philippines": "PHL",
}

df = pd.DataFrame(columns=["country", "date", "urate"])
for country, seriesid in tqdm(dict_fred_urate.items()):
    df_sub = get_data_from_fred(id=seriesid, api=fred_api_key)
    df_sub["country"] = country
    df_sub = df_sub.rename(columns={"value": "urate"})
    df = pd.concat([df, df_sub], axis=0)
df["quarter"] = pd.to_datetime(df["date"]).dt.to_period("Q")
df.loc[df["urate"] == ".", "urate"] = np.nan
df["urate"] = df["urate"].astype("float") 
df = df.groupby(["country", "quarter"])["urate"].mean().reset_index()
df = df.dropna()
df = df.sort_values(by=["country", "quarter"], ascending=[True, True])
df = df.reset_index(drop=True)

# %%
# II --- Wrangle

# %%
# III --- Output
# Save processed output
df["quarter"] = df["quarter"].astype("str")
df.to_parquet(path_data + "urate_quarterly" + ".parquet")

# %%
# X --- Notify
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%