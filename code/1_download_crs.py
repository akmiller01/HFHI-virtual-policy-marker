from oda_reader import download_crs

# Download small subset first
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
crs_sector_data.to_csv('input/crs_2014_2023_purpose_codes.csv')

# Next, all disbursements
crs_data = download_crs(
    start_year=2014, 
    end_year=2023, 
    filters={
        'flow_type': 'D',
        'price_base': 'Q',
        'microdata': True
    }, 
    dataflow_version='1.3'
)
crs_data.to_csv('large_input/crs_2014_2023.csv')
