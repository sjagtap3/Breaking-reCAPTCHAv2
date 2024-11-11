import csv

def parse_log_file(log_file):
    attempts = []
    attempt_data = {
        "classification_type": 0,
        "type_2": 0,
        "total_attempts": 0
    }

    with open(log_file, 'r') as f:
        for line in f:
            if "************** New attempt started **************" in line:
                attempt_data = {
                    "classification_type": 0,
                    "type_2": 0,
                    "total_attempts": 0
                }
            elif "Classification type captcha encountered" in line:
                attempt_data["classification_type"] += 1
            elif "Type 2 captcha encountered" in line:
                attempt_data["type_2"] += 1
            elif "Captcha is now solved, total attempts:" in line:
                attempt_data["total_attempts"] = int(line.split(":")[-1].strip())
                attempts.append(attempt_data)

    return attempts

def save_to_csv(data, csv_file):
    fieldnames = ["classification_type", "type_2", "total_attempts"]
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


log_file = 'captcha_log_file_baseline.log'
csv_file = 'captcha_data_baseline_novpn_stats.csv'

data = parse_log_file(log_file)
save_to_csv(data, csv_file)