import pandas as pd
import seaborn as sns
import numpy as np


def test(df):
    return np.log(df["gdpPercap"])

gapminder = pd.read_csv("https://raw.githubusercontent.com/OHI-Science/data-science-training/master/data/gapminder.csv")
gapminder = gapminder.assign(
  log_gdp=test,
  log_pop=lambda df: np.log(df["pop"]),
  decade=lambda df: np.floor(df["year"] / 10) * 10
)
print()