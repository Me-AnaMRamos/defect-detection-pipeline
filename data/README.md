## Data Directory

- `raw/`: Original, immutable dataset files (not tracked by git)
- `processed/`: Cleaned and transformed data used for training and evaluation

Raw data must never be modified directly.

## Dataset

This project uses the MVTec Anomaly Detection (MVTec AD) dataset.

### Download
The dataset must be downloaded manually from:
https://www.mvtec.com/company/research/datasets/mvtec-ad

### Expected directory structure

After extraction, place the dataset at:

data/raw/mvtec_ad/

Example:
data/raw/mvtec_ad/bottle/train/good/
data/raw/mvtec_ad/bottle/test/broken_large/

### Git policy
Raw and processed data are excluded from version control.
Only metadata, preprocessing scripts, and documentation are committed.
