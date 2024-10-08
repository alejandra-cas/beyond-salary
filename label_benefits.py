import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

even_sample = pd.read_parquet('data/salary_sample_body.parquet.gzip')


career_dev = "training programs", "education assistance","tuition reimbursement", "tuition assistance", "mentorship", "career growth", "professional development", "leadership development", "career advancement", "leadership training", "growth opportunities", "personal development", "education reimbursement"
wlb = "paid time off", "PTO", "extra vacation", "flexible vacation", "mental health day", "work-life balance", "work/life balance","generous time-off", "generous time off", "paid vacation", "paid holidays", "holiday pay"
wellbeing = "wellness stipend", "health benefits", "mental health support", "gym membership", "wellness program", "well-being stipend", "wellness programs", "mental health benefits", "employee well-being", "health care benefits"
recognition = "employee of the month", "long-term rewards", "recognition program", "peer recognition", "service awards", "employee recognition"
family = "parental leave", "childcare assistance", "childcare support", "childcare", "family support", "flexible maternity leave", "paternity leave", "maternity leave", "paid family leave", "child care discount"
culture = "collaborative environment", "diversity and inclusion", "team culture", "creative freedom", "values-driven", "inclusive culture", "diverse team", "inclusive environment"
keywords_list = [career_dev,wlb,wellbeing, recognition, family, culture]

def check_benefits(df, keywords, benefit):
    pattern = r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b'
    df[benefit] = df['BODY'].str.contains(pattern, case=False, regex=True)
    return df
    
def plot_counts(df, benefit):
    counts = df.groupby(['HAS_AI_SKILLS',benefit]).size().reset_index()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='HAS_AI_SKILLS', y=0, hue=benefit, data=counts)
    # rename x axis labels
    labels = ['Non-AI Role', 'AI Role']
    ax.set_xticklabels(labels)
    ax.set_xlabel(None)
    ax.set_ylabel('Jobs')
    
check_benefits(even_sample, career_dev, 'CAREER_DEV')
check_benefits(even_sample, wlb, 'WLB')
check_benefits(even_sample, wellbeing, 'WELLBEING')
check_benefits(even_sample, recognition, 'RECOGNITION')
check_benefits(even_sample, family, 'FAMILY')
check_benefits(even_sample, culture, 'CULTURE')

print(even_sample.head())

even_sample.to_parquet('data/salary_sample_body_benefits.parquet.gzip', compression='gzip')

