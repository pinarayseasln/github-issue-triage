import pandas as pd

# Ham veriyi okuyoruz
df = pd.read_csv("data/raw/github_issues.csv")

print(df.head())
print(df.shape)
print(df["labels"].value_counts().head(20))