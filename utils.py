import os
import random

import csv 

def log_eval_result(entry_dict, csv_path="results_log.csv"):
    """
    Append a row of evaluation results to a CSV. If file does not exist, create it with headers.

    Args:
        entry_dict (dict): Dictionary where key = column name, value = data to log.
        csv_path (str): Path to the CSV file.
    """
    print('wring to csv', csv_path)
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=entry_dict.keys(), delimiter='\t')
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(entry_dict)