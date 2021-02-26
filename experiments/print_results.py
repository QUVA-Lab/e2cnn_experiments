
import glob
import os.path

import sqlite3
import pandas as pd


def retrieve_logs(path: str) -> pd.DataFrame:
    conn = sqlite3.connect(path)
    logs = pd.read_sql_query("select * from logs;", conn)
    conn.close()
    
    return logs


for folder in glob.iglob("./results/*/"):
    dataset = os.path.basename(os.path.normpath(folder))
    
    print('###########################################################################################################')
    print(f"DATASET: {dataset}")
    
    exps = []
    
    for db in glob.iglob(os.path.join(folder, "*.db")):
        model = os.path.splitext(os.path.basename(db))[0]
        
        # if model.endswith("confusion"):
        #     continue
        
        logs = retrieve_logs(db)
        
        logs = logs[logs.split == "test"]
        last_iter = logs.iteration.max()
        logs = logs[logs.iteration == last_iter].groupby("seed").first()
        accuracies = logs.accuracy
        
        errors = 100.0 - 100.0 * accuracies
        
        e = "{:<55} | min: {:.5f}; mean: {:.5f}; std: {:.5f} | samples: {}".format(model, errors.min(), errors.mean(), errors.std(), len(accuracies))
        exps.append(e)
    
    for exp in sorted(exps):
        print(exp)
        
        
    


