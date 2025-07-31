import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import time

# input_path = input("Please enter the input parquet file path: ")
# output_path = input("Please enter the output parquet file path: ")

input_path = '../data/us_10m_nointernship_2018_2024_body.parquet.gzip'
output_path = '../data/remote_kw_labels_2024.parquet.gzip'
# even_sample = pd.read_parquet('data/salary_sample_body.parquet.gzip')
load_time = time.time()
print("Loading the input parquet file...")
if input_path[-3:] == 'csv:':
    even_sample = pd.read_csv(input_path)
elif input_path[-3:] == 'zip':
    even_sample = pd.read_parquet(input_path)

print("Time taken to load the input parquet file: ", time.time()-load_time)
print(even_sample.columns)
even_sample = even_sample[['ID','BODY']]
empty_body_count = even_sample['BODY'].isna().sum() + even_sample['BODY'].str.strip().eq("").sum()
print("Number of empty body text: ", empty_body_count)

def clean_text(text):
    if isinstance(text, str):  # Ensure the value is a string
        text = text.strip()  # Remove leading and trailing spaces
        text = ' '.join(text.split())  # Replace multiple spaces with a single space
        text = text.replace('\n', ' ')  # Remove newline characters
        text = text.lower()
    return text

print("cleaning body")
clean_time = time.time()
# Apply the cleaning function to the Remote_KW column
even_sample['BODY'] = even_sample['BODY'].apply(clean_text)
print("Time taken to clean body: ", time.time()-clean_time)

remote_keywords = [
    "fully remote", "100% remote", "work from home", "remote role", "work remotely", 
    "remote eligible", "open to remote", "remote work environment", "remote work policy", 
    "remote and onsite", "virtual role", "remote first", "home office", "telecommute", 
    "distributed team", "remote-first", "remote-friendly", "remote options", "virtual work", "work from home", 
    "remote position", "wfh", "Remote - us", "telework", "home office", "(remote)", "remote work flexibility", "remote work: hybrid", 
    "remote: yes", "location: remote", "remote usa", "remote flexibility", "(remote", "- remote", "remote-flexibility", 
    "remote,", "\nremote\n", "\n remote \n", "remotely", "us-remote", "remote work eligible", "remote - united states", 
    "remote - work at home", "can be remote", "remote within the us", "remote in north america", "remote available" ", remote", 
    "remote full-time", "us remote", "#li-remote", "open to remote",
    "hybrid work", "split time between office and remote", "office and remote options", 
    "partially remote", "hybrid position", "3 days remote", "hybrid workplace", 
    "some remote days", "in-office and remote", "hybrid onsite", "remote or in-office", "work-from-home days", "mix of working in the office and from home", "#li-hybrid", "hybrid remote"
]

remote_to_exclude = [
    "not considering remote", "in-office only", 
    "remote monitoring", "remote sensing", "remote access systems", "remote control", 
    "remote diagnostics", "remote delivery", "#li-onsite", "onsite job", "work from home not available", "telework:no", 
    "remote: no", "100% on-site", "work at home option: No",
    "remotely: no", "remote: n", "remotely: n", "telework: no", "remote: * no", "remotely: * no",
    "remotely piloted", "data remotely", "remote type on-site", "interviewed remotely", "work remotely: * no", 
    "remote desktop", "must be able to work on-site", "not applicable for 100% remote", "remote testing", "remotely upgrading", "remote usability",
    "remote site", "remote machine", "remote iot", "no remote", "work remotely no", "remotely:no", "remotely? n", "remotely no", "remote areas", 
    "remote access", "remote position? no", "remotely tucked away", "supporting remote", "remotely sensed"
]


def check_benefits(even_sample, keywords, benefit, exclusions=None):
    # Create inclusion and exclusion patterns
    include_pattern = r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b'
    exclude_pattern = r'\b(' + '|'.join(map(re.escape, exclusions)) + r')\b' if exclusions else None
    
    def match_benefit(text):
        # Check if the text contains any inclusion keywords
        if not isinstance(text, str) or pd.isna(text) or text.strip() == "":
            return False  # Return False if no text is present to search in  
             
        includes = bool(re.search(include_pattern, text, re.IGNORECASE))
        if not includes:
            return False
        
        if exclusions:
            # Then, check if the text contains any exclusion keywords
            excludes = bool(re.search(exclude_pattern, text, re.IGNORECASE))
            
            # If inclusion is found but exclusion also exists, check carefully
            if includes and excludes:
                # Allow "True" if the exclusion is found but inclusion still exists in a separate context
                # (i.e., don't automatically set False just because exclusion exists)
                include_matches = list(re.finditer(include_pattern, text, re.IGNORECASE))
                exclude_matches = list(re.finditer(exclude_pattern, text, re.IGNORECASE))
                
                # Ensure inclusion and exclusion aren't in the same region of text
                for exc in exclude_matches:
                    for inc in include_matches:
                        # If the exclusion keyword exactly matches or overlaps with the inclusion keyword, skip it
                        if inc.start() <= exc.start() < inc.end() or inc.start() <= exc.end() <= inc.end():
                            return False
                return True
        
        return includes

    # Apply the match_benefit function to each row
    even_sample[benefit] = even_sample['BODY'].apply(match_benefit)
    
    return even_sample

    
def label_remote_hybrid(row):
    body = row['BODY']
    if not isinstance(body, str) or pd.isna(body) or body.strip() == "":
        return 0  # Return False if no text is present to search in  
    if any(kw in body for kw in remote_to_exclude):
        return 0 
    elif any(kw in body for kw in remote_keywords):
        return 1    
    return 0

# # Apply the labeling function
# print("Checking benefits...")    
# start = time.time()
# even_sample["REMOTE_KW"] = even_sample.apply(label_remote_hybrid, axis=1)
# print("time taken to check benefits: ", time.time()-start)
# print("exporting")
# export = even_sample.drop(columns=['BODY'])
# export.to_parquet(output_path, compression='gzip')
    


print("Checking benefits method 2...")    
# print time taken to check benefits
import time
start = time.time()
# print start time
print("Start time: ", start)
print("Checking remote work...")
check_benefits(even_sample, remote_keywords, 'REMOTE_KW', remote_to_exclude)
print("time taken to check benefits: ", time.time()-start)

print(even_sample.head())

print("Saving to parquet...")
# drop body column and export
save_time = time.time()
even_sample.drop(columns=['BODY']).to_parquet(output_path, compression='gzip')
# print("saving with body")

# even_sample.to_parquet(output_path, compression='gzip')
print("Time taken to save to parquet: ", time.time()-save_time)



