# Time Series Checkpoint

This checkpoint is designed to test your knowledge of time series analysis and modeling.

Specifically, this will cover:

* Using `pandas` to manipulate time series data
* Plotting time series data
* Modeling time series data with an ARMA model

## Data Understanding

The following dataset comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction). It includes energy usage data as well as temperature and humidity data.

The relevant columns for your analysis will be:

* `date`: A string representing the timestamp of measurements taken every 10 minutes
* `lights`: An integer representing energy use of light fixtures in the house in Wh


```python
# Run this cell without changes
import pandas as pd

df = pd.read_csv("energy_data.csv")
df
```

## 1. Create a `Series` Object for Analysis

As noted previously, we do not need all of the columns of `df`. Create a `pandas` `Series` object called `light_ts` which has an index of type `DatetimeIndex` generated based on the `date` column of `df` and data from the values of the `lights` column of `df`.

***Hint:*** The `pd.to_datetime` function ([documentation here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html)) can convert strings representing dates into datetimes


```python
# Replace None with appropriate code
light_ts = None
# your code here
raise NotImplementedError
light_ts
```


```python
# light_ts should be a Series
assert type(light_ts) == pd.Series

# light_ts should have the same number of records as df
assert light_ts.shape[0] == df.shape[0]

# The index of light_ts should be composed of datetimes
assert type(light_ts.index) == pd.DatetimeIndex

# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```

## 2. Downsample Time Series Data to Find a Daily Total

Currently this dataset has recorded the amount of energy used every hour. We want to analyze the amount of energy used every day.

Create a `Series` called `daily_ts` which contains the data from `light_ts` downsampled using the frequency string for 1 **day**, then aggregated using the **sum** of daily energy use by the lights.

***Hint:*** Here is some relevant documentation:

* See [this page](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.resample.html#pandas.Series.resample) for information on the method used for upsampling and downsampling
* See [this page](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects) for the available frequency strings
* See [this page](https://pandas.pydata.org/docs/reference/resampling.html#computations-descriptive-stats) for the available aggregate functions to use after resampling


```python
# Replace None with appropriate code
daily_ts = None
# your code here
raise NotImplementedError
daily_ts
```


```python
# daily_ts should be a Series
assert type(daily_ts) == pd.Series

# daily_ts should have fewer records than light_ts
assert len(daily_ts) < len(light_ts)

# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```


```python
# Run this cell without changes
daily_ts.plot(ylabel="Daily energy use of lights (Wh)");
```

## 3. Check for Stationarity

Is this `daily_ts` time series stationary? You can answer this by interpreting the graph above, or using a statistical test ([documentation here](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html)) with an alpha of 0.05. Assign your answer to `is_stationary`.


```python
# Replace None with appropriate code
is_stationary = None
# your code here
raise NotImplementedError
is_stationary
```


```python
# is_stationary should be True or False
assert (is_stationary == True or is_stationary == False)

# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```

## 4. Find The Weekly Rolling Average of Time Series Data

Create a `Series` called `rolling_avg_ts` that represents the **weekly (7-day)** rolling **mean** of daily energy usage.

***Hint:*** See [this documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.rolling.html) for more information on calculating a rolling average.

(Note that it is expected that you'll see some NaN values at the beginning, when there have been fewer than 7 days to average.)


```python
# Replace None with appropriate code
rolling_avg_ts = None
# your code here
raise NotImplementedError
rolling_avg_ts
```


```python
# rolling_avg_ts should be a Series
assert type(rolling_avg_ts) == pd.Series

# rolling_avg_ts should have the same number of records as daily_ts
assert len(rolling_avg_ts) == len(daily_ts)

# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```

In the cell below, we plot the raw daily data, the 7-day moving average, and the difference between the raw daily data and the moving average.


```python
# Run this cell without changes

import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
import numpy as np

differenced_ts = daily_ts - rolling_avg_ts
differenced_ts.dropna(inplace=True)
differenced_transformed_ts = np.sqrt(daily_ts) - np.sqrt(rolling_avg_ts)
differenced_transformed_ts.dropna(inplace=True)

fig, axes = plt.subplots(ncols=3, figsize=(16,4))

axes[0].plot(daily_ts, color="gray", label="Daily energy use", )
axes[0].plot(rolling_avg_ts, color="blue", label="7-day moving average")
axes[1].plot(differenced_ts, color="green", label="Differenced")
axes[2].plot(differenced_transformed_ts, label="Differenced and transformed")

locator = AutoDateLocator()
formatter = ConciseDateFormatter(locator)

for ax in axes:
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.legend()
```

## 5. Choose a Dataset and Build an AR(I)MA Model

Based on the plots above, choose the most-stationary time series data out of:

* `daily_ts`
* `differenced_ts`
* `differenced_transformed_ts`

And plug it into an AR(I)MA model ([documentation here](https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.html))

You can use any order `(p, d, q)` that you want, so long as it doesn't produce a warning message.


```python
# Replace None with appropriate code
from statsmodels.tsa.arima.model import ARIMA

model = None
# your code here
raise NotImplementedError

res = model.fit()
res.summary()
```


```python
# model should be an ARIMA model
assert type(model) == ARIMA

# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```
