### Classifications of machine learning features
### For use across files
### Follows organisation & order of thesis table, though exact wording may differ

import numpy as np
import pandas as pd

# Features

# feature: [feature category, coding]
# coding types: nominal, ordinal, binary, continuous

feature_dict = {

    
    ### Ophthalmic
    ##############################
    
    'IOPg pre-treatment': ['ophthalmic', 'continuous'],
    'IOPg pre-treatment inter-eye difference': ['ophthalmic', 'continuous'],
    'Corneal hysteresis': ['ophthalmic', 'continuous'],
    'Corneal hysteresis inter-eye difference': ['ophthalmic', 'continuous'],
    'Corneal resistance factor': ['ophthalmic', 'continuous'],
    'Spherical equivalent': ['ophthalmic', 'continuous'],
    #'Self-reported myopia': ['ophthalmic', 'binary'],

    
    ### Demographic
    ##############################
    
    'Polygenic risk score': ['demographic', 'continuous'],
    'Age at initial assesement': ['demographic', 'continuous'],
    'Sex': ['demographic', 'binary'],
    'Ethnicity': ['demographic', 'binary'],
    'PM2.5 exposure': ['demographic', 'continuous'],
    'Urban residence': ['demographic', 'binary'],
    'Townsend deprivation index': ['demographic', 'continuous'],
    'Total household income': ['demographic', 'ordinal'],
    'Education': ['demographic', 'ordinal'],
    'Private healthcare utilisation': ['demographic', 'ordinal'],


    # Systemic
    ##############################

     # Neuro & psychiatric
    'Dementia (baseline)': ['systemic', 'binary'],
    'Migraine (baseline)': ['systemic', 'binary'],
    'Anxiety disorder (baseline)': ['systemic', 'binary'],

     # Hearing
    'Hearing difficulty (self-reported)': ['systemic', 'binary'],
    'Tinnitus frequency (self-reported)': ['systemic', 'ordinal'],
    'Speech reception threshold': ['systemic', 'continuous'],
    'Conductive and sensorineural hearing loss (baseline)': ['systemic', 'binary'],
    'Other hearing loss (baseline)': ['systemic', 'binary'],

    # Vascular
    'Systolic blood pressure': ['systemic', 'continuous'],
    'Diastolic blood pressure': ['systemic', 'continuous'],
    'Arterial stiffness index': ['systemic', 'continuous'],
    'Hypertension (baseline)': ['systemic', 'binary'],
    'Hypotension (baseline)': ['systemic', 'binary'],
    'Peripheral vascular disease (baseline)': ['systemic', 'binary'],

    # Diabetes mellitus
    'HbA1c': ['systemic', 'continuous'],
    'Plasma glucose': ['systemic', 'continuous'],
    'Diabetes mellitus (baseline)': ['systemic', 'binary'],

    # Lipids
    'Triglycerides': ['systemic', 'continuous'],
    'Total cholesterol': ['systemic', 'continuous'],
    'HDL': ['systemic', 'continuous'],
    'LDL': ['systemic', 'continuous'],
    'Dyslipidaemia (baseline)': ['systemic', 'binary'],

    # Other cardio-metabolic related
    'Sleep apnoea (baseline)': ['systemic', 'binary'],
    'Atrial fibrillation or flutter (baseline)': ['systemic', 'binary'],

    # Renal
    'eGFR serum creatinine': ['systemic', 'continuous'],
    'Albumin-creatinine ratio': ['systemic', 'continuous'],
    'Chronic kidney disease (baseline)': ['systemic', 'binary'],

    # Endocrine
    'Plasma oestradiol': ['systemic', 'continuous'],
    'Plasma testosterone': ['systemic', 'continuous'],
    'Hypothyroidism (baseline)': ['systemic', 'binary'],
    'Thyrotoxicosis (baseline)': ['systemic', 'binary'],


     # Gastrointestinal
    'Poor oral health': ['systemic', 'binary'],
    'Gingivitis or periodontitis (baseline)': ['systemic', 'binary'],
    'Helicobacter pylori infection (baseline)': ['systemic', 'binary'],
    'Irritable bowel syndrome (baseline)': ['systemic', 'binary'],

    # Auto-immune
    'Psoriasis (baseline)': ['systemic', 'binary'],
    'Sjogren syndrome (baseline)': ['systemic', 'binary'],
    'Rheumatoid arthritis (baseline)': ['systemic', 'binary'],

    # Immune-related
    'Rosacea (baseline)': ['systemic', 'binary'],
    'COPD (baseline)': ['systemic', 'binary'],
    'Asthma (baseline)': ['systemic', 'binary'],
    'Atopic dermatitis (baseline)': ['systemic', 'binary'],
    'Vasomotor or allergic rhinitis (baseline)': ['systemic', 'binary'],
    'Chronic sinusitis (baseline)': ['systemic', 'binary'],

    # Inflammatory markers
    'Systemic immune inflammation index': ['systemic', 'continuous'],
    'C-reactive protein': ['systemic', 'continuous'],

    # Oxidative stress markers
    'Plasma urate': ['systemic', 'continuous'],
    'Plasma total bilirubin': ['systemic', 'continuous'],
    'Plasma albumin': ['systemic', 'continuous'],

     # Medications
    'Metformin': ['systemic', 'binary'],
    'Statin': ['systemic', 'binary'],
    'Beta blocker': ['systemic', 'binary'],
    'Calcium channel blocker': ['systemic', 'binary'],
    'ACE inhibitor': ['systemic', 'binary'],
    'Angiotensin receptor blocker': ['systemic', 'binary'],
    'Diuretic': ['systemic', 'binary'],
    'SSRI': ['systemic', 'binary'],
    'SNRI': ['systemic', 'binary'],


    ### Lifestyle
    ##############################
    

    'Exercise (summed MET minutes per week)': ['lifestyle', 'continuous'],
    'Body mass index': ['lifestyle', 'continuous'],
    'Plasma Vitamin D': ['lifestyle', 'continuous'],
    'Diet score': ['lifestyle', 'continuous'],

    'Vitamin C supplementation': ['lifestyle', 'binary'],
    'Multivitamin supplementation': ['lifestyle', 'binary'],
    'Glucosamine supplementation': ['lifestyle', 'binary'],
    'Iron supplementation': ['lifestyle', 'binary'],
    'Selenium supplementation': ['lifestyle', 'binary'],
    'Calcium supplementation': ['lifestyle', 'binary'],
    
    'Salt added to food': ['lifestyle', 'ordinal'],
    'Urinary sodium-creatinine ratio': ['lifestyle', 'continuous'],
    'Caffeinated coffee drinker': ['lifestyle', 'binary'],
    'Tea intake': ['lifestyle', 'continuous'], #'Tea intake': ['lifestyle', 'ordinal'],
    'Alcohol intake': ['lifestyle', 'ordinal'],
    'Current smoking frequency': ['lifestyle', 'ordinal'],
    'Past smoking frequency': ['lifestyle', 'ordinal'],
    
    'Normal sleep duration': ['lifestyle', 'binary'],
    'Insomnia frequency': ['lifestyle', 'ordinal'],
    'Snoring': ['lifestyle', 'binary'],
    'Daytime sleeping frequency': ['lifestyle', 'ordinal'],
}


### Define feature vars for models

ODSL_features = pd.DataFrame.from_dict(data=feature_dict, orient='index', columns=['category', 'coding_type'])
ODSL_features = ODSL_features.reset_index().rename(columns={'index': 'feature'})

ophthalmic_features = ODSL_features[ODSL_features['category'].isin(['ophthalmic'])].reset_index(drop=True)
demographic_features = ODSL_features[ODSL_features['category'].isin(['demographic'])].reset_index(drop=True)
systemic_features = ODSL_features[ODSL_features['category'].isin(['systemic'])].reset_index(drop=True)
lifestyle_features = ODSL_features[ODSL_features['category'].isin(['lifestyle'])].reset_index(drop=True)

ODS_features = ODSL_features[ODSL_features['category'].isin(['ophthalmic', 'demographic', 'systemic'])].reset_index(drop=True)
OD_features = ODSL_features[ODSL_features['category'].isin(['ophthalmic', 'demographic'])].reset_index(drop=True)
SL_features = ODSL_features[ODSL_features['category'].isin(['systemic', 'lifestyle'])].reset_index(drop=True)

DSL_features = ODSL_features[ODSL_features['category'].isin(['demographic', 'systemic', 'lifestyle'])].reset_index(drop=True)


### Features for minimised model with RFECV, n=15

minimal_features_rfecv = ODSL_features[ODSL_features['feature'].isin([
    'IOPg pre-treatment', 'IOPg pre-treatment inter-eye difference',
    'Corneal hysteresis', 'Corneal hysteresis inter-eye difference',
    'Corneal resistance factor', 'Spherical equivalent',
    'Polygenic risk score', 'Age at initial assesement', 'Ethnicity',
    'Townsend deprivation index', 'Diastolic blood pressure', 'HbA1c',
    'Total cholesterol', 'eGFR serum creatinine',
    'Urinary sodium-creatinine ratio'
])]