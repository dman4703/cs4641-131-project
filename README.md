# Mean Reversion Trading with Machine Learning
CS4641 Group 131

Devon O'Quinn, Shayali Patel, Nicholas Nitsche, Julien Perez, Mutimu Njenga

**Report:** <https://github.gatech.edu/pages/doquinn3/cs4641-131-project/>

## env setup
```bash
# create env
conda env create -f ml_project.yml
conda activate ml_project
```

## Project Structure

### Data
`/data/raw/`: Raw tick-by-tick trade data CSV files organized by ticker symbol (20 stocks, 5 trading days each)

`/data/excel/`: Original Excel files containing static tick data exports

`/data/clean/`: Cleaned and preprocessed parquet files with NBBO alignment and microstructure flags
- `/data/clean/metadata.parquet`: Metadata and statistics for cleaned datasets
- `/data/clean/metadata.csv`: CSV version of metadata
- `/data/clean/cleaning.log`: Data cleaning process logs

`/data/processed/`: Feature-engineered volume bar data ready for ML models
- `/data/processed/processing_metadata.parquet`: Processing pipeline statistics
- `/data/processed/processing_metadata.csv`: CSV version of processing metadata
- `/data/processed/cv_metadata.parquet`: Cross-validation fold metadata
- `/data/processed/cv_metadata.csv`: CSV version of cross-validation metadata
- `/data/processed/pipeline_config.yaml`: Pipeline configuration snapshot
- `/data/processed/pipeline.log`: Processing pipeline execution logs

`/data/gmm/`: GMM model outputs and trading candidates
- `/data/gmm/gmm_candidates_5pct.parquet`: GMM-flagged candidates (5% threshold)
- `/data/gmm/gmm_candidates_10pct.parquet`: GMM-flagged candidates (10% threshold)

### Configuration Files
`/ml_project.yml`: Conda environment configuration with all project dependencies

`/config/`: Configuration files for data processing pipeline
- `/config/preprocessing.yaml`: Data cleaning policies (NBBO handling, timezone, quality gates)
- `/config/processing.yaml`: Feature engineering and volume bar configuration
- `/config/condition_codes.csv`: Trade condition code mappings

### Source Code
`/src/dataCleaning/`: Data cleaning and preprocessing modules
- `/src/dataCleaning/parseToCSV.py`: Excel to CSV conversion utility
- `/src/dataCleaning/clean_data.py`: data cleaning script (NBBO alignment, duplicate removal, condition filtering)
- `/src/dataCleaning/inspect_cleaned_data.py`: Data quality inspection tools
- `/src/dataCleaning/test_clean_data.py`: Unit tests for data cleaning pipeline

`/src/dataProcessing/`: Feature engineering and bar construction
- `/src/dataProcessing/volume_bars.py`: Volume bar aggregation implementation
- `/src/dataProcessing/feature_engineering.py`: feature computation
- `/src/dataProcessing/labeling.py`: labeling script
- `/src/dataProcessing/cross_validation.py`: cross-validation utilities
- `/src/dataProcessing/process_pipeline.py`: Processing script
- `/src/dataProcessing/config_processing.py`: Configuration handling for processing pipeline

`/src/gmm/`: Gaussian Mixture Model for anomaly-based candidate selection
- `/src/gmm/data.py`: Data loading and preprocessing utilities
- `/src/gmm/model.py`: GMM training and inference
- `/src/gmm/evaluate.py`: Model evaluation and visualization

### Notebooks
`/notebooks/dataCleaning.ipynb`: Data cleaning pipeline demonstration and validation

`/notebooks/dataEDA.ipynb`: Exploratory data analysis of raw tick data

`/notebooks/dataProcessing.ipynb`: Feature engineering and volume bar construction walkthrough

`/notebooks/determineBar.ipynb`: Analysis for optimal volume bar size selection

`/notebooks/gmm.ipynb`: GMM model training, hyperparameter tuning, and evaluation

### Models
`/models/gmm.joblib`: Trained Gaussian Mixture Model

`/models/gmm_scaler.joblib`: Feature scaler fitted on training data

`/models/gmm_config.json`: Model configuration and performance metrics

### Reports & Documentation
`/docs/`: Project website and documentation
- `/docs/index.html`: Report
- `/docs/assets/img/`: Visualization assets and plots for website