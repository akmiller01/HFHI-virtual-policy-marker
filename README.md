# HFHI-virtual-policy-marker
NLP techniques to build a virtual adequate housing policy marker

## Code Overview

### 1.0_download_crs.py

This script downloads and processes CRS (Creditor Reporting System) data. It performs the following tasks:

1. **Download a small subset of CRS data**: 
   - Inputs: 
     - `start_year`: 2014
     - `end_year`: 2023
     - `filters`: Specific sector codes ('16030', '16040'), flow type ('D'), price base ('Q'), and microdata flag.
     - `dataflow_version`: '1.3'
   - Output: Saves the data to `input/crs_2014_2023_purpose_codes.csv` if it doesn't already exist.

2. **Download all disbursements for each year from 2014 to 2023**:
   - Inputs: 
     - `start_year`: Year in the range 2014 to 2023
     - `end_year`: Same as `start_year`
     - `filters`: Flow type ('D'), price base ('Q'), and microdata flag.
     - `dataflow_version`: '1.3'
   - Output: Saves the data for each year to `large_input/crs_<year>.csv` if it doesn't already exist.

3. **Collate all yearly disbursements into a single file**:
   - Inputs: Yearly CSV files from `large_input/crs_<year>.csv`
   - Output: Concatenates all yearly data into a single file `large_input/crs_2014_2023.csv` if it doesn't already exist.

### 1.1_sector_code_analysis.R

This script analyzes the CRS data by various dimensions such as category, year, donor, and recipient. It performs the following tasks:

1. **Setup and load necessary libraries**: Installs and loads required R packages.
2. **Load and filter data**: Reads the CRS data from `input/crs_2014_2023_purpose_codes.csv` and filters it for disbursements in constant prices.
3. **Analyze data by category**: Aggregates data by year and category, and generates a bar chart saved as `output/sector_category_year.png`. Also saves the aggregated data to `output/sector_category_year.csv`.
4. **Analyze data by year**: Aggregates data by year and sector, and generates a bar chart saved as `output/sector_year.png`. Also saves the aggregated data to `output/sector_year.csv`.
5. **Analyze data by donor**: Aggregates data by donor, generates a bar chart for the top 10 donors saved as `output/sector_by_donor.png`, and saves the aggregated data to `output/sector_by_donor.csv`.
6. **Analyze data by donor type**: Merges CRS data with donor type reference, aggregates data by donor type, generates a bar chart saved as `output/sector_by_donor_type.png`, and saves the aggregated data to `output/sector_by_donor_type.csv`.
7. **Analyze data by recipient**: Aggregates data by recipient, generates a bar chart for the top 10 recipients saved as `output/sector_by_recipient.png`, and saves the aggregated data to `output/sector_by_recipient.csv`.
8. **Analyze data by recipient income group**: Merges CRS data with income group information, aggregates data by income group, generates a bar chart saved as `output/sector_by_income.png`, and saves the aggregated data to `output/sector_by_income.csv`.

### 1.2_collate_sector_xlsx.R

This script collates the results of the sector code analysis into a single Excel file. It performs the following tasks:

1. **Setup and load necessary libraries**: Loads required R packages.
2. **Load CSV files**: Reads all CSV files from the `output` directory that match the pattern `sector_.*.csv`.
3. **Create Excel workbook**: Creates a new Excel workbook and adds a worksheet for each CSV file.
4. **Insert data and images**: Writes the data from each CSV file into the corresponding worksheet and inserts any associated images (e.g., bar charts) into the worksheet.
5. **Save workbook**: Saves the Excel workbook as `output/current_sector_analysis.xlsx`.

### 2.0_upload_crs_dataset.py

This script prepares and uploads the collated CRS dataset to the Hugging Face Hub. It performs the following tasks:

1. **Setup and load necessary libraries**: Loads required Python packages and the Hugging Face Hub token.
2. **Load and preprocess data**: Reads the collated CRS data from `large_input/crs_2014_2023.csv`, creates a unique text column by combining project title, short description, and long description, and removes duplicates and blank entries.
3. **Count tokens**: Counts the number of tokens in each text entry using a tokenizer.
4. **Upload dataset**: Uploads the processed dataset to the Hugging Face Hub under the repository `alex-miller/crs-2014-2023`.

### 2.1a_embedding_retrieval.py

This script generates embeddings for the CRS dataset and retrieves similar entries based on a query. It performs the following tasks:

1. **Setup and load necessary libraries**: Loads required Python packages and the Hugging Face Hub token.
2. **Load dataset**: Reads the processed CRS dataset from the Hugging Face Hub.
3. **Generate embeddings**: Uses a pre-trained model to generate embeddings for each text entry in the dataset.
4. **Retrieve similar entries**: Given a query, retrieves and ranks similar entries from the dataset based on their embeddings.
5. **Output results**: Outputs the top similar entries for the given query.

### 2.1b_embedding_filter.py

This script filters the CRS dataset using embeddings and keyword matching to identify relevant entries. It performs the following tasks:

1. **Setup and load necessary libraries**: Loads required Python packages and the Hugging Face Hub token.
2. **Load dataset**: Reads the processed CRS dataset from the Hugging Face Hub.
3. **Load keywords**: Reads a list of keywords from `input/keywords.csv`.
4. **Apply keyword matching**: Identifies entries containing the specified keywords.
5. **Define target condition**: Marks entries as relevant if they match the sector codes or contain the keywords.
6. **Train model**: Trains an XGBoost classifier to predict relevance based on the dataset features.
7. **Evaluate model**: Evaluates the model's performance using accuracy, F1 score, recall, and precision metrics.
8. **Save results**: Saves the confusion matrix as an image and uploads the filtered dataset to the Hugging Face Hub.

### 2.2_gpt_label.py

This script uses GPT-4 to label the CRS dataset with additional classifications. It performs the following tasks:

1. **Setup and load necessary libraries**: Loads required Python packages and the Hugging Face Hub token.
2. **Load dataset**: Reads the filtered CRS dataset from the Hugging Face Hub.
3. **Estimate tokens**: Estimates the number of tokens required for labeling and warns the user about the potential cost.
4. **Label data**: Uses GPT-4 to label each entry in the dataset with additional classifications based on the text and sector code.
5. **Save results**: Saves the labeled dataset to the Hugging Face Hub under the repository `alex-miller/crs-2014-2023-housing-labeled-gpt`.

### 2.2_ollama_label.py

This script uses the Ollama model to label the CRS dataset with additional classifications. It performs the following tasks:

1. **Setup and load necessary libraries**: Loads required Python packages and the Hugging Face Hub token.
2. **Load dataset**: Reads the filtered CRS dataset from the Hugging Face Hub.
3. **Label data**: Uses the Ollama model to label each entry in the dataset with additional classifications based on the text and sector code.
4. **Save results**: Saves the labeled dataset to the Hugging Face Hub under the repository `alex-miller/crs-2014-2023-housing-labeled-phi4`.

### 2.3_merge.py

This script merges the original CRS dataset with the labeled dataset to create a comprehensive dataset. It performs the following tasks:

1. **Setup and load necessary libraries**: Loads required Python packages.
2. **Load original dataset**: Reads the original CRS data from `large_input/crs_2014_2023.csv`.
3. **Create text column**: Combines project title, short description, and long description into a unique text column.
4. **Load labeled dataset**: Reads the labeled dataset from the Hugging Face Hub.
5. **Merge datasets**: Merges the original dataset with the labeled dataset based on the unique text column.
6. **Save results**: Saves the merged dataset to `large_output/crs_2014_2023_phi4_labeled.csv`.

### 3.0_virtual_policy_marker_analysis.R

This script performs an analysis of the virtual policy marker for adequate housing. It performs the following tasks:

1. **Setup and load necessary libraries**: Installs and loads required R packages.
2. **Load and preprocess data**: Reads the merged CRS data from `large_output/crs_2014_2023_phi4_labeled.csv`.
3. **Filter data**: Filters the data to include only relevant entries based on various classifications such as Homelessness, Transitional, Incremental, Social, Market, and Sector code.
4. **Analyze data by year**: Aggregates data by year and classification, and generates bar charts saved as `output/virtual_year.png` and `output/virtual_year_percent.png`. Also saves the aggregated data to `output/virtual_year.csv` and `output/virtual_year_percent.csv`.
5. **Analyze data by urban/rural**: Aggregates data by year and urban/rural classification, and generates bar charts saved as `output/urban_rural_year_percent.png`. Also saves the aggregated data to `output/urban_rural_year_percent.csv`.
