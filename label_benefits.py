import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import time

input_path = input("Please enter the input parquet file path: ")
output_path = input("Please enter the output parquet file path: ")

# even_sample = pd.read_parquet('data/salary_sample_body.parquet.gzip')
load_time = time.time()
print("Loading the input parquet file...")
even_sample = pd.read_parquet(input_path)
print("Time taken to load the input parquet file: ", time.time()-load_time)

empty_body_count = even_sample['BODY'].isna().sum() + even_sample['BODY'].str.strip().eq("").sum()
print("Number of empty body text: ", empty_body_count)

# career_dev = "training programs", "education assistance","tuition reimbursement", "tuition assistance", "mentorship", "career growth", "professional development", "leadership development", "career advancement", "leadership training", "growth opportunities", "personal development", "education reimbursement", "opportunities to grow and develop"
edu_assistance = "education assistance","tuition reimbursement", "tuition assistance", "education reimbursement"
wlb = "paid time off", "PTO", "extra vacation", "flexible vacation", "mental health day", "work-life balance", "work/life balance","generous time-off", "generous time off", "paid vacation", "paid holidays", "holiday pay", "vacation days", "company holiday", "sick leave", "vacation time", "paid days off", "paid flexible holidays"
leave = "paid time off", "PTO", "extra vacation", "flexible vacation", "mental health day", "generous time-off", "generous time off", "paid vacation", "paid holidays", "holiday pay", "vacation days", "company holiday", "sick leave", "vacation time", "paid days off", "paid flexible holidays"
wellbeing = "wellness stipend", "mental health support", "gym membership", "wellness program", "well-being stipend", "wellness programs", "mental health benefits", "employee well-being", "mental wellness support", "wellness and global well-being", "wellness stipend"
health_wellbeing = "wellness stipend", "health benefits", "mental health support", "gym membership", "wellness program", "well-being stipend", "wellness programs", "mental health benefits", "employee well-being", "health care benefits"
# recognition = "employee of the month", "long-term rewards", "recognition program", "peer recognition", "service awards", "employee recognition"
# family = "parental leave", "childcare assistance", "childcare support", "childcare", "family support", "flexible maternity leave", "paternity leave", "maternity leave", "paid family leave", "child care discount", "paid caregiver/parental", "paid parental", "paid new parent leave"
parental_leave = "parental leave", "family support", "flexible maternity leave", "paternity leave", "maternity leave", "paid family leave", "paid caregiver/parental", "paid parental", "paid new parent leave", "paid bonding leave", "parental bonding leave"
culture = "diversity and inclusion", "team culture", "creative freedom", "values-driven", "inclusive culture", "diverse team", "inclusive environment", "value diversity", "diversity, equity", "culture of diversity", "diversity, inclusion", "inclusion and diversity", "commitment to diversity", "diversity, inclusion", "diversity, equity and inclusion", "diversity is respected", "inclusive diversity", "workforce diversity", "embracing diversity", "value diversity", "values diversity", "committed to diversity", "equity and diversity", "promoting diversity", "celebrate diversity", "encourage diversity", "support diversity", "diversity in the workplace", "equity, inclusion"
# keywords_list = [career_dev,wlb,wellbeing, health_wellbeing, recognition, family, parental_leave, culture]

# export keywords list as pickle
# import pickle
# with open('data/keywords_list.pkl', 'wb') as f:
#     pickle.dump(keywords_list, f)

career_dev_to_exclude = ["youth leadership development", "provide mentorship", "professional development expertise","planning professional development", "forecasting growth opportunities", "teen leadership development", "develop training programs", "professional development experience", "training programs as required", "providing mentorship", "providing and encouraging mentorship", "implementing training programs", "provide leadership and mentorship", "provide mentorship", "identify growth opportunities", "manage the Symbotic Tuition Reimbursement"]
tuition_to_exclude = ["manage the Symbotic Tuition Reimbursement"]
wellbeing_to_exclude = ["health benefits companies", "wellness program coordinator", "familiarity with mental health support", "health benefits administration"]
recognition_to_exclude = ["managing employee recognition"]
family_to_exclude = ["accredited childcare program", "maui family support services", "teaching assistant - childcare", "experience in childcare", "childcare state licensing", "experience with children"]
culture_to_exclude = ["maintaining a collaborative environment"]

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

    
# def plot_counts(even_sample, benefit):
#     counts = even_sample.groupby(['HAS_AI_SKILLS',benefit]).size().reset_index()
#     plt.figure(figsize=(10, 6))
#     ax = sns.barplot(x='HAS_AI_SKILLS', y=0, hue=benefit, data=counts)
#     # rename x axis labels
#     labels = ['Non-AI Role', 'AI Role']
#     ax.set_xticklabels(labels)
#     ax.set_xlabel(None)
    ax.set_ylabel('Jobs')

print("Checking benefits...")    
# print time taken to check benefits
import time
start = time.time()
# print start time
print("Start time: ", start)
# check_benefits(even_sample, career_dev, 'CAREER_DEV', exclusions=career_dev_to_exclude)
# check_benefits(even_sample, edu_assistance, 'EDU_ASSISTANCE', exclusions=tuition_to_exclude)
# print time
print("Time taken for edu assistance: ", time.time()-start)

# check time
wlb_time = time.time()
print("Checking work-life balance...")
# check_benefits(even_sample, wlb, 'WLB')
print("Time taken for work-life balance: ", time.time()-wlb_time)

print("Checking leave...")
check_benefits(even_sample, leave, 'LEAVE')
even_sample.rename(columns={'LEAVE':'PAID_LEAVE'}, inplace=True)

# check_benefits(even_sample, wellbeing, 'WELLBEING', wellbeing_to_exclude)
print("Checking health & wellbeing...")
# check_benefits(even_sample, health_wellbeing, 'HEALTH_WELLBEING', wellbeing_to_exclude)
# check_benefits(even_sample, recognition, 'RECOGNITION', recognition_to_exclude)
# check_benefits(even_sample, family, 'FAMILY', family_to_exclude)
print("Checking parental leave...")
# check_benefits(even_sample, parental_leave, 'PARENTAL_LEAVE')
print("Checking culture...")
# check_benefits(even_sample, culture, 'CULTURE')
print("time taken to check benefits: ", time.time()-start)

print(even_sample.head())

print("Saving to parquet...")
save_time = time.time()
even_sample.to_parquet(output_path, compression='gzip')
print("Time taken to save to parquet: ", time.time()-save_time)

# drop body column and export
even_sample.drop(columns=['BODY']).to_parquet('data/us_10m_nointernship_ai_skills_benefits.parquet.gzip', compression='gzip')

