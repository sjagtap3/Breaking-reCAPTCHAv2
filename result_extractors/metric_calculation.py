import pandas as pd
import numpy as np

def calculate_statistics(csv_file):
    df = pd.read_csv(csv_file)

    for column in df.columns:
        non_zero_values = [x for x in df[column] if x != 0]
    
        mean = np.mean(non_zero_values)
        median = np.median(non_zero_values)
        # mean = df[column].mean()
        # median = df[column].median()
        # mode = df[column].mode()
        # quartiles = df[column].quantile([0.25, 0.5, 0.75]).values

        print(f"Statistics for {column}:")
        print(f"Mean: {mean}")
        print(f"Median: {median}")
        # print(f"Mode: {mode}")
        # print(f"Quartiles: Q1={quartiles[0]}, Q2={quartiles[1]}, Q3={quartiles[2]}")
        print()

csv_file = 'captcha_log_file_baseline_vpn_stats.csv'
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


# Our best model
# Statistics for classification_type:
# Mean: 13.72
# Median: 1.0
# Mode: 1
# Quartiles: Q1=1.0, Q2=1.0, Q3=5.25

# Statistics for type_2:
# Mean: 3.42
# Median: 1.5
# Mode: 0
# Quartiles: Q1=0.0, Q2=1.5, Q3=2.0

# Statistics for total_attempts:
# Mean: 17.16
# Median: 3.0
# Mode: 1
# Quartiles: Q1=2.0, Q2=3.0, Q3=9.5


# With best models & VPN
# Statistics for classification_type:
# Mean: 3.761904761904762
# Median: 1.0
# Mode: 0    1
# Name: classification_type, dtype: int64
# Quartiles: Q1=0.0, Q2=1.0, Q3=1.75

# Statistics for type_2:
# Mean: 0.9761904761904762
# Median: 1.0
# Mode: 0    0
# Name: type_2, dtype: int64
# Quartiles: Q1=0.0, Q2=1.0, Q3=1.75

# Statistics for total_attempts:
# Mean: 4.738095238095238
# Median: 2.0
# Mode: 0    1
# Name: total_attempts, dtype: int64
# Quartiles: Q1=1.0, Q2=2.0, Q3=5.0
