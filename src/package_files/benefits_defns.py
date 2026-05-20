# benefits = ['CAREER_DEV', 'WLB', 'WELLBEING', 'RECOGNITION', 'FAMILY', 'CULTURE']
benefits2 = [
    "CAREER_DEV",
    "WLB",
    "HEALTH_WELLBEING",
    "RECOGNITION",
    "FAMILY",
    "CULTURE",
]
benefits3 = [
    "CAREER_DEV",
    "EDU_ASSISTANCE",
    "WLB",
    "WELLBEING",
    "HEALTH_WELLBEING",
    "RECOGNITION",
    "FAMILY",
    "PARENTAL_LEAVE",
    "CULTURE",
]
benefits3_labels = [
    "Career Development",
    "Education Assistance",
    "Work-Life Balance",
    "Wellbeing",
    "Health and Wellbeing",
    "Recognition",
    "Family",
    "Parental Leave",
    "Culture",
]
benefits4 = [
    "EDU_ASSISTANCE",
    "PAID LEAVE",
    "HEALTH_WELLBEING",
    "PARENTAL_LEAVE",
    "CULTURE",
    "REMOTE_KW",
]
benefits_labels_map = {
    "EDU_ASSISTANCE": "Tuition Assistance",
    "PAID LEAVE": "Paid Leave",
    "HEALTH_WELLBEING": "Health and Wellbeing",
    "PARENTAL_LEAVE": "Parental Leave",
    "CULTURE": "Workplace Culture",
    "wfh_wham": "Remote Work",
    "REMOTE_KW": "Remote Work",
    "S_FLEX_WORK": "Flexible Work Schedules",
    "S_PROF_DEV": "Professional Development",
    "S_HEALTH_WELLNESS": "Health & Wellness Programs",
    "S_REMOTE": "Remote Work (Structured)",
}
benefits4_labels = [
    "Tuition Assistance",
    "Paid Leave",
    "Health and Wellbeing",
    "Parental Leave",
    "Workplace Culture",
    "Remote Work",
]
benefits5 = [
    "EDU_ASSISTANCE",
    "PAID LEAVE",
    "HEALTH_WELLBEING",
    "PARENTAL_LEAVE",
    "CULTURE",
]

# Extended set including structured-field benefits
benefits6 = benefits4 + ["S_FLEX_WORK", "S_PROF_DEV", "S_HEALTH_WELLNESS", "S_REMOTE"]
benefits6_labels = benefits4_labels + [
    "Flexible Work Schedules",
    "Professional Development",
    "Health & Wellness Programs",
    "Remote Work (Structured)",
]

region = "STATE_NAME"
industry = "NAICS_2022_2_NAME"
education = "MIN_EDULEVELS_NAME"
year = "YEAR"
occupation = "SOC_MAJOR_GROUP"
experience = "EXPERIENCE_BUCKET"
firm = "COMPANY"
firm_name = "COMPANY_NAME"

benefit_colors = {
    "EDU_ASSISTANCE": "#41afaa",
    "PAID LEAVE": "#466eb4",
    "HEALTH_WELLBEING": "#e6a532",
    "PARENTAL_LEAVE": "#00a0e1",
    "CULTURE": "#d7642c",
    "wfh_wham": "#af4b91",
    "REMOTE_KW": "#c765a6",
    "S_FLEX_WORK": "#7b9f35",
    "S_PROF_DEV": "#5c6bc0",
    "S_HEALTH_WELLNESS": "#ef6c00",
    "S_REMOTE": "#8e24aa",
}

benefit_colors_2 = {
    "EDU_ASSISTANCE": ["#41afaa", "#b3dfdd"],
    "PAID LEAVE": ["#466eb4", "#b5c5e1"],
    "HEALTH_WELLBEING": ["#e6a532", "#f5dbad"],
    "PARENTAL_LEAVE": ["#00a0e1", "#99d9f3"],
    "CULTURE": ["#d7642c", "#efc1ab"],
    "wfh_wham": ["#af4b91", "#dfb7d3"],
    "REMOTE_KW": ["#af4b91", "#dfb7d3"],
    "S_FLEX_WORK": ["#7b9f35", "#c5d89a"],
    "S_PROF_DEV": ["#5c6bc0", "#b3bae0"],
    "S_HEALTH_WELLNESS": ["#ef6c00", "#f5b980"],
    "S_REMOTE": ["#8e24aa", "#cfa3d9"],
}
