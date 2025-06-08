from oda_reader import download_crs_file
import pandas as pd

# All years
all_filename = 'large_input/crs_2014_2023.csv'
all_dataframes = []
for year in range(2014, 2024):
    year_filename = 'large_input/crs_{}.csv'.format(year)
    crs_data = download_crs_file(year=str(year))
    crs_data = crs_data[crs_data['USD_Disbursement_Defl'] > 0]
    all_dataframes.append(crs_data)
    crs_data.to_csv(year_filename)

# Collate
concatenated_df = pd.concat(all_dataframes, ignore_index=True)
concatenated_df.to_csv(all_filename)

# Small sector subset
sector_filename = 'input/crs_2014_2023_purpose_codes.csv'
crs_sector_data = concatenated_df[concatenated_df['PurposeCode'].isin([16030, 16040])]
crs_sector_data.to_csv(sector_filename)
