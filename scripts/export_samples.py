import pandas as pd
import yaml
from pathlib import Path


def load_sample_paths():
    """Load processed/sample paths from config.yaml.

    Falls back to repo-local defaults if config.yaml is not found.
    """
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / "config.yaml"

    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["data"]
        processed_dir = Path(cfg["processed_dir"])
        if not processed_dir.is_absolute():
            processed_dir = repo_root / processed_dir
    else:
        print(
            "Warning: config.yaml not found, using default processed paths. "
            "Copy config.example.yaml to config.yaml to configure."
        )
        processed_dir = repo_root / "data" / "processed"

    sample_dir = processed_dir.parent / "small_samples"
    return {
        "input_path": processed_dir / "data_v1.parquet",
        "salary_output_path": sample_dir / "2024_salary_sample.parquet.gzip",
        "nosalary_output_path": sample_dir / "2024_nosalary_sample.parquet.gzip",
    }


paths = load_sample_paths()
paths["salary_output_path"].parent.mkdir(parents=True, exist_ok=True)
all_data = pd.read_parquet(paths["input_path"])
# all_data = all_data[all_data['QUARTER'] != '2024Q3']
# all_data_body = pd.read_parquet('../data/us_10m_nointernship_ai_skills_body.parquet.gzip')
# Exports

print("finished loading data")

## Random Sample with Salary

salary_data = all_data[all_data['SALARY'].notnull()]
# salary_wham_data = salary_data[salary_data['wfh_wham'].notnull()]
len(salary_data)
salary_ai = salary_data[salary_data['AI ROLE'] == True]
salary_no_ai = salary_data[salary_data['AI ROLE'] == False]
salary_ai_sample = salary_ai.sample(n=10000, random_state=42)
salary_no_ai_sample = salary_no_ai.sample(n=10000, random_state=42)
salary_sample_all = pd.concat([salary_ai_sample, salary_no_ai_sample])
# salary_sample_all = salary_sample_all.merge(body, left_on='ID', right_on='ID', how='left')
len(salary_sample_all)
salary_sample_all.to_parquet(paths["salary_output_path"], compression='gzip')


## Random Sample without Salary
# all_data_wham = all_data[all_data['wfh_wham'].notnull()]
all_data_ai = all_data[all_data['AI ROLE'] == True]
all_data_no_ai = all_data[all_data['AI ROLE'] == False]
all_data_ai_sample = all_data_ai.sample(n=10000, random_state=42)
all_data_no_ai_sample = all_data_no_ai.sample(n=10000, random_state=42)
all_data_sample_all = pd.concat([all_data_ai_sample, all_data_no_ai_sample])
# all_data_sample_all = all_data_sample_all.merge(body, left_on='ID', right_on='ID', how='left')
all_data_sample_all.to_parquet(paths["nosalary_output_path"], compression='gzip')
# print("saving just benefits all data")
# all_data.drop(columns=['BODY']).to_parquet('data/us_10m_nointernship_ai_skills_benefits.parquet.gzip', compression='gzip')
