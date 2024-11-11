import pandas as pd

def calculate_statistics(csv_file):
    df = pd.read_csv(csv_file)

    for column in df.columns:
        mean = df[column].mean()
        median = df[column].median()
        mode = df[column].mode().values[0]
        quartiles = df[column].quantile([0.25, 0.5, 0.75]).values

        print(f"Statistics for {column}:")
        print(f"Mean: {mean}")
        print(f"Median: {median}")
        print(f"Mode: {mode}")
        print(f"Quartiles: Q1={quartiles[0]}, Q2={quartiles[1]}, Q3={quartiles[2]}")
        print()

csv_file = 'captcha_data_v11_novpn_stats.csv'
calculate_statistics(csv_file)


# captcha_data_baseline_novpn_stats.csv
# Statistics for classification_type:
# Mean: 24.42
# Median: 1.0
# Mode: 1
# Quartiles: Q1=1.0, Q2=1.0, Q3=11.5

# Statistics for type_2:
# Mean: 1.58
# Median: 1.0
# Mode: 0
# Quartiles: Q1=0.0, Q2=1.0, Q3=2.0

# Statistics for total_attempts:
# Mean: 26.06
# Median: 2.5
# Mode: 1
# Quartiles: Q1=1.0, Q2=2.5, Q3=14.25


# captcha_data_v11_novpn_stats.csv
# Statistics for classification_type:
# Mean: 21.22
# Median: 1.0
# Mode: 1
# Quartiles: Q1=1.0, Q2=1.0, Q3=3.5

# Statistics for type_2:
# Mean: 1.58
# Median: 2.0
# Mode: 0
# Quartiles: Q1=0.0, Q2=2.0, Q3=2.0

# Statistics for total_attempts:
# Mean: 22.86
# Median: 3.0
# Mode: 1
# Quartiles: Q1=1.0, Q2=3.0, Q3=5.75