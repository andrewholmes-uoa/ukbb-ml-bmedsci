import numpy as np
import pandas as pd


# Diagnosis constructor
# Read diagnoses from occurrence fields, inpatient, and GP data
# (See data_extraction.ipynb for example)
######################################################


### Occurrences

invalid_occurrences_dates = [
    pd.to_datetime('01/01/1900', format='%d/%m/%Y'), # no event date
    pd.to_datetime('01/01/1901', format='%d/%m/%Y'), # precedes DOB
    pd.to_datetime('02/02/1902', format='%d/%m/%Y'), # matches DOB
    pd.to_datetime('03/03/1903', format='%d/%m/%Y'), # follows DOB, but is in year of birth
    pd.to_datetime('09/09/1909', format='%d/%m/%Y'), # dated in the future meaning system default / placeholder
    pd.to_datetime('07/07/2037', format='%d/%m/%Y') # dated in the future meaning system default / placeholder
]

# occurrence_data_col: the occurrence column to read (e.g. 131286)
# baseline_feid_set: a set() containing f.eid items. baseline cases of the outcome will be added to here.
# incident_feid_set: as above, for incident cases.
# feid_dates_df: a DataFrame containing occurrence diagnosis dates; use to calculate earliest diagnosis date across multiple occurrence fields.
def extract_occurrence(df, occurrence_data_col, baseline_feid_set, incident_feid_set, feid_dates_df=None):
    # Baseline cases
    any_baseline_dx = df[occurrence_data_col] <= df['f.53.0.0']
    baseline_dx_valid = any_baseline_dx & ~(df[occurrence_data_col].isin(invalid_occurrences_dates))
    baseline_feid_set.update(df[baseline_dx_valid]['f.eid'])

    # Incident cases
    any_incident_dx = df[occurrence_data_col] > df['f.53.0.0']
    incident_dx_valid = any_incident_dx & ~(df[occurrence_data_col].isin(invalid_occurrences_dates))
    incident_feid_set.update(df[incident_dx_valid]['f.eid'])

    # Invalid date counts
    n_baseline_invalid = any_baseline_dx.sum() - baseline_dx_valid.sum()
    n_incident_invalid = any_incident_dx.sum() - incident_dx_valid.sum()

    # Earliest date across multiple occurrences
    if feid_dates_df is not None:
        dates_df = df[baseline_dx_valid | incident_dx_valid][['f.eid', occurrence_data_col]]
        dates_df = dates_df.rename(columns={occurrence_data_col: 'earliest_occurrence_diagnosis_date'})
        feid_dates_df = pd.concat((feid_dates_df, dates_df), axis=0)

    return n_baseline_invalid, n_incident_invalid, feid_dates_df


### Primary care

invalid_gp_dates = [
    pd.to_datetime('01/01/1901', format='%d/%m/%Y'), # precedes DOB
    pd.to_datetime('02/02/1902', format='%d/%m/%Y'), # matches DOB
    pd.to_datetime('03/03/1903', format='%d/%m/%Y'), # follows DOB, but is in year of birth
    pd.to_datetime('07/07/2037', format='%d/%m/%Y') # dated in the future meaning system default / placeholder
]

def extract_gp(merged_gp_events_df, read_v2_codes, read_v3_codes, baseline_feid_set, incident_feid_set):
    # All cases
    any_dx = (merged_gp_events_df['read_2'].isin(read_v2_codes)) | (merged_gp_events_df['read_3'].isin(read_v3_codes))

    # Baseline cases
    any_baseline_dx = any_dx & (merged_gp_events_df['event_dt'] <= merged_gp_events_df['f.53.0.0'])
    baseline_dx_valid = any_baseline_dx & ~(merged_gp_events_df['event_dt'].isin(invalid_gp_dates))
    baseline_feid_set.update(merged_gp_events_df[baseline_dx_valid]['f.eid'])

    # Incident cases
    any_incident_dx = any_dx & (merged_gp_events_df['event_dt'] > merged_gp_events_df['f.53.0.0'])
    incident_dx_valid = any_incident_dx & ~(merged_gp_events_df['event_dt'].isin(invalid_gp_dates))
    incident_feid_set.update(merged_gp_events_df[incident_dx_valid]['f.eid'])

    # Earliest date of GP diagnosis
    earliest_date_gp = merged_gp_events_df[baseline_dx_valid | incident_dx_valid].groupby('f.eid')['event_dt'].min()

    # Invalid date counts
    n_baseline_invalid = any_baseline_dx.sum() - baseline_dx_valid.sum()
    n_incident_invalid = any_incident_dx.sum() - incident_dx_valid.sum()

    return earliest_date_gp, n_baseline_invalid, n_incident_invalid


### Inpatient

# There are no invalid dates for inpatient (unlike occurrences and primary care)

def process_inpatient_matrix(df, diagnosis_col_list, date_col_list, positive_values_list, baseline_feid_set, incident_feid_set, feid_dates_df=None):
    for col_idx in range(len(diagnosis_col_list)):
        dx_col = diagnosis_col_list[col_idx]
        date_col = date_col_list[col_idx]
        
        any_dx = df[dx_col].isin(positive_values_list) 
        baseline_dx = any_dx & (df[date_col] <= df['f.53.0.0'])
        incident_dx = any_dx & (df[date_col] > df['f.53.0.0'])

        baseline_feid_set.update(df[baseline_dx]['f.eid'])
        incident_feid_set.update(df[incident_dx]['f.eid'])

        # Add to first inpatient dict for keeping track of earliest diagnoses
        # NOTE: may have multiple instances: to find earliest instance, group by f.eid and get min()
        if feid_dates_df is not None:
            for index, row in df[['f.eid', date_col]][any_dx].iterrows():
                feid_dates_df.loc[len(feid_dates_df)] = [row['f.eid'], row[date_col]]


def extract_inpatient(df, icd10_codes, icd9_codes, baseline_feid_set, incident_feid_set, feid_dates_df=None):
    ICD10_inpatient_dx_cols = df.columns[df.columns.str.startswith('f.41270.0')]
    ICD10_inpatient_date_cols = df.columns[df.columns.str.startswith('f.41280.0')]
    
    ICD9_inpatient_dx_cols = df.columns[df.columns.str.startswith('f.41271.0')]
    ICD9_inpatient_date_cols = df.columns[df.columns.str.startswith('f.41281.0')]

    feid_dates_df = pd.DataFrame(columns=['f.eid', 'earliest_inpatient_diagnosis_date'])
    
    # Extract ICD10
    if len(icd10_codes) > 0:
        process_inpatient_matrix(
            df,
            ICD10_inpatient_dx_cols,
            ICD10_inpatient_date_cols,
            icd10_codes,
            baseline_feid_set,
            incident_feid_set,
            feid_dates_df
        )

    # Extract ICD9
    if len(icd9_codes) > 0:
        process_inpatient_matrix(
            df,
            ICD9_inpatient_dx_cols,
            ICD9_inpatient_date_cols,
            icd9_codes,
            baseline_feid_set,
            incident_feid_set,
            feid_dates_df
        )

    # Return earliest date of inpatient diagnosis
    return feid_dates_df.groupby('f.eid')['earliest_inpatient_diagnosis_date'].min()



# Get case, training & test definitions from shared data file
######################################################

def merge_case_and_diagnosis_cols(df_to_merge, file_name='v3_stratified_bindx_dx_and_train_test_split.pkl'):
    # Read file with train/test split and glaucoma definitions
    defs_df = pd.read_pickle('/mnt/shared_folders/eResearch_glaucoma_project/andrewholmes2024/shared_data/'+file_name)
    cols = defs_df.columns

    # Drop columns from the df to merge, where we are going to replace them
    # avoids double-ups upon merging
    cols_to_drop = np.delete(cols, np.argwhere(cols=='f.eid')) # avoid dropping f.eid
    df_edited = df_to_merge.drop(cols_to_drop, errors='ignore', axis=1)

    # merge dataframes
    merged_df = pd.merge(defs_df[cols], df_edited, on='f.eid', how='outer')
    
    return merged_df


    
# Missing feature analysis
######################################################

def get_missing_feature_stats(df, feature_dict, df_save_path=None):
    overall_dict = {}
    for feature_set_name, features in feature_dict.items():
        feature_set_dict = {}
        
        total_n = len(df)
        feature_set_dict['Num features'] = len(features)

        n_missing_any = df[features].isna().any(axis=1).sum()
        feature_set_dict['Participants missing ≥1 feature (%)'] = f'{n_missing_any} ({((n_missing_any / total_n) * 100):0.2f})'
        
        n_missing_5_or_more = (df[features].isna().sum(axis=1) >= 5).sum()
        feature_set_dict['Participants missing ≥5 features (%)'] = f'{n_missing_5_or_more} ({((n_missing_5_or_more / total_n) * 100):0.2f})'
        
        n_missing_10_or_more = (df[features].isna().sum(axis=1) >= 10).sum()
        feature_set_dict['Participants missing ≥10 features (%)'] = f'{n_missing_10_or_more} ({((n_missing_10_or_more / total_n) * 100):0.2f})'
        
        feature_set_dict['Mean features missing (std)'] = f'{(df[features].isna().sum(axis=1)).mean():0.2f} ({(df[features].isna().sum(axis=1)).std():0.2f})'
        
        median = (df[features].isna().sum(axis=1)).median()
        lower = (df[features].isna().sum(axis=1)).quantile(0.25)
        upper = (df[features].isna().sum(axis=1)).quantile(0.75)
        feature_set_dict['Median features missing (IQR)'] = f'{median:0.2f} ({lower:0.2f}, {upper:0.2f})'
        feature_set_dict['Max features missing'] = f'{(df[features].isna().sum(axis=1)).max()}'

        overall_dict[feature_set_name] = feature_set_dict

    df = pd.DataFrame.from_dict(overall_dict, orient='columns')

    if df_save_path != None:
        df.to_csv(df_save_path, header=True, index=True, mode='w')

    return df



def get_missing_feature_list(df, feature_list, df_save_path=None):
    total_n = len(df)
    feature_missing_dict = {}
    for feature in feature_list:
        n_missing = df[feature].isna().sum()
        n_missing_percent = (n_missing / total_n) * 100
        feature_missing_dict[feature] = {
            'n': n_missing,
            'Participants missing feature (%)': f'{n_missing} ({n_missing_percent:0.2f})',
        }

    ordered_dict = {k: v for k, v in sorted(feature_missing_dict.items(), key=lambda x: x[1]['n'], reverse=True)}
    df = pd.DataFrame.from_dict(ordered_dict, orient='index')
    df = df.drop(columns=['n'])

    if df_save_path != None:
        df.to_csv(df_save_path, header=True, index=True, mode='w', index_label='Feature')

    return df