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

csv_file = 'captcha_data_v11_novpn.csv'
calculate_statistics(csv_file)


# captcha_data_baseline_novpn.csv
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
# Mean: 35.74
# Median: 35.0
# Mode: 22
# Quartiles: Q1=27.25, Q2=35.0, Q3=44.75



# captcha_data_v11_novpn.csv'
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
# Mean: 33.42
# Median: 29.5
# Mode: 28
# Quartiles: Q1=25.25, Q2=29.5, Q3=40.75
