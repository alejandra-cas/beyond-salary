import pandas as pd 

print("Loading the 10m sample...")
data = pd.read_parquet('data/us_10m_nointernship_ai_skills_body.parquet.gzip')
print("getting NAs")
# get nas for has ai skills
ai_na = data[data['AI ROLE'].isna()]
print("Number of NAs: ", len(ai_na))
print("loading skill IDs")
# Classify Jobs With AI Skills
with open('data/ai_skill_ids.pkl', 'rb') as f:
    ai_skill_ids = pickle.load(f)
ai_skill_ids
# Function to check if any AI skill ID is in the job's skills
def has_ai_skills(skill_list, ai_skill_ids):
    skill_list = eval(skill_list)  # Convert string representation of list to actual list
    return any(skill_id in skill_list for skill_id in ai_skill_ids)
ai_na['Has AI Skills'] = usdf.swifter.progress_bar(True).apply(lambda x: has_ai_skills(x['SKILLS'], ai_skill_ids), axis=1)

data.loc[ai_na.index, 'AI ROLE'] = ai_na['AI ROLE']

data.to_parquet('data/us_10m_nointernship_ai_skills_body.parquet.gzip', compression='gzip')