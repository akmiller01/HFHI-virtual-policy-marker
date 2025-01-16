import os
from oda_reader import download_crs
import pandas as pd

# Download small subset first
sector_filename = 'input/crs_2014_2023_purpose_codes.csv'
if not os.path.exists(sector_filename):
    crs_sector_data = download_crs(
        start_year=2014, 
        end_year=2023, 
        filters={
            'sector': ['16030', '16040'],
            'flow_type': 'D',
            'price_base': 'Q',
            'microdata': True
        }, 
        dataflow_version='1.3'
    )
    crs_sector_data.to_csv(sector_filename)

# Next, all disbursements
for year in range(2014, 2024):
    year_filename = 'large_input/crs_{}.csv'.format(year)
    if not os.path.exists(year_filename):
        crs_data = download_crs(
            start_year=year, 
            end_year=year, 
            filters={
                'flow_type': 'D',
                'price_base': 'Q',
                'microdata': True
            }, 
            dataflow_version='1.3'
        )
        crs_data.to_csv(year_filename)

# Collate
all_filename = 'large_input/crs_2014_2023.csv'
if not os.path.exists(all_filename):
    all_dataframes = []

    for year in range(2014, 2024):
        year_filename = 'large_input/crs_{}.csv'.format(year)
        df = pd.read_csv(year_filename, encoding='utf-8')
        all_dataframes.append(df)
    concatenated_df = pd.concat(all_dataframes, ignore_index=True)
    concatenated_df.to_csv(all_filename)
