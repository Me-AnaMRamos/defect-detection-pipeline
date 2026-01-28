import pandas as pd

INDEX_CSV = "data/processed/dataset_index.csv"

df = pd.read_csv(INDEX_CSV)

# Count per split
counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)

print("Class counts per split:")
print(counts)

# Ratios (defective prevalence)
ratios = counts.div(counts.sum(axis=1), axis=0)
print("\nClass ratios per split:")
print(ratios)
