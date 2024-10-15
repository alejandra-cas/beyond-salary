import pandas as pd 


# get nas for has ai skills
ai_na = data[data['Has AI Skills'].isna()]

# Classify Jobs With AI Skills
with open('../data/ai_skill_ids.pkl', 'rb') as f:
    ai_skill_ids = pickle.load(f)
ai_skill_ids
# Function to check if any AI skill ID is in the job's skills
def has_ai_skills(skill_list, ai_skill_ids):
    skill_list = eval(skill_list)  # Convert string representation of list to actual list
    return any(skill_id in skill_list for skill_id in ai_skill_ids)
ai_na['Has AI Skills'] = usdf.swifter.progress_bar(True).apply(lambda x: has_ai_skills(x['SKILLS'], ai_skill_ids), axis=1)

data.loc[ai_na.index, 'Has AI Skills'] = ai_na['Has AI Skills']