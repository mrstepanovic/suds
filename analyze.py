"""
PUT DOCS HERE
"""

import bdb_utils
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt

# statics
API_KEY = "c7638c9e8a98b169c092481ba409bd9f"
URL = "http://api.brewerydb.com/v2/beers/"
params = dict(key=API_KEY, withBreweries='Y')

# Download beer data into pandas dataframe.
beer_data = bdb_utils.download_beer_data(URL, params, 500).drop_duplicates()

# Change quantitative datatypes to numeric.
beer_data_clean[['ABV','IBU', 'SRM', 'OG']] = beer_data_clean[['ABV','IBU', 'SRM', 'OG']].apply(pd.to_numeric)

# Plot counts of top 10 beer styles.
pd.value_counts(beer_data_clean['Style']).head(n=10).plot(kind='barh')

# Save top 10 beer styles, by count, to a dataframe.
counts = pd.value_counts(beer_data_clean['Style']).head(n=10)
popular_beers = beer_data_clean[beer_data_clean['Style'].isin(counts.index)]




