# benefits = ['CAREER_DEV', 'WLB', 'WELLBEING', 'RECOGNITION', 'FAMILY', 'CULTURE']
benefits2 = ['CAREER_DEV', 'WLB', 'HEALTH_WELLBEING', 'RECOGNITION', 'FAMILY', 'CULTURE']
benefits3 = ['CAREER_DEV', 'EDU_ASSISTANCE', 'WLB', 'WELLBEING', 'HEALTH_WELLBEING', 'RECOGNITION', 'FAMILY', 'PARENTAL_LEAVE', 'CULTURE']
benefits3_labels = ['Career Development', 'Education Assistance', 'Work-Life Balance', 'Wellbeing', 'Health and Wellbeing','Recognition', 'Family', 'Parental Leave', 'Culture']
benefits4 = ['EDU_ASSISTANCE','PAID_LEAVE','HEALTH_WELLBEING', 'PARENTAL_LEAVE', 'CULTURE', 'REMOTE_KW']
benefits_labels_map = {'EDU_ASSISTANCE': 'Tuition Assistance', 'PAID_LEAVE': 'Paid Leave', 'HEALTH_WELLBEING': 'Health and Wellbeing', 'PARENTAL_LEAVE': 'Parental Leave', 'CULTURE': 'Workplace Culture', 'wfh_wham': 'Remote Work', 'REMOTE_KW': 'Remote Work'}
benefits4_labels = ['Tuition Assistance', 'Paid Leave', 'Health and Wellbeing', 'Parental Leave', 'Workplace Culture', 'Remote Work']
benefits5 = ['EDU_ASSISTANCE','PAID_LEAVE','HEALTH_WELLBEING', 'PARENTAL_LEAVE', 'CULTURE']

region = 'STATE_NAME'
industry = 'NAICS_2022_2_NAME'
education = 'MIN_EDULEVELS_NAME'
year = 'YEAR'
occupation = 'SOC_2021_2_NAME'
experience = 'EXPERIENCE_BUCKET'

benefit_colors = {
    'EDU_ASSISTANCE': '#41afaa',
    'PAID_LEAVE': '#466eb4',
    'HEALTH_WELLBEING': '#e6a532',
    'PARENTAL_LEAVE': '#00a0e1',
    'CULTURE': '#d7642c',
    'wfh_wham': '#af4b91', 
    'REMOTE_KW': '#c765a6'
}