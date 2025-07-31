path = "../data/us_10m_nointernship_2018_2024_benefits.parquet.gzip"

import os 
# check if path exists
if not os.path.exists(path):
    raise FileNotFoundError(f"The file {path} does not exist.")
