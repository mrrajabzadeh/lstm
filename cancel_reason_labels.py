# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:39:52 2024

@author: mraja
"""


# Import required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential, Input
from tensorflow.keras.layers import Masking, LSTM, TimeDistributed, Layer, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from imblearn.over_sampling import SMOTE
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import Workbook
from tensorflow.keras.regularizers import l2
from sklearn.utils import resample
from tensorflow.keras import backend as K
import shap
from tensorflow.keras.models import Model

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# -----------------------------------------------------------------------------
# Part 1: Merging Data and Calculating Some Simple Statistics
# -----------------------------------------------------------------------------

# Define the path to the directory containing the files
file_path_pattern = r'E:/Data/Merged/*.txt'

# Use glob to get a list of all files matching the pattern
all_files = glob.glob(file_path_pattern)

# Initialize an empty list to hold individual DataFrames
dfs = []

# Loop over the list of files and read each one into a DataFrame
for file in all_files:
    df = pd.read_csv(file, delimiter='|')
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Define desired visit types
desired_visit_types = [
    'BLOOD DRAW', 'ESTABLISHED PATIENT', 'ESTABLISHED PATIENT ON TX',
    'TREATMENT CHEMO', 'TREATMENT', 'NEW CHEMO', 'BLOOD PRODUCT'
]

# Filter the DataFrame for desired visit types
df = merged_df[merged_df['VISIT_TYPE'].isin(desired_visit_types)]

# Convert ENCOUNTER_DTTM to datetime
df['DATE'] = pd.to_datetime(df['ENCOUNTER_DTTM'], format='mixed', errors='coerce', dayfirst=True)

# Sort DataFrame by PT_ID and DATE
df = df.sort_values(by=['PT_ID', 'DATE'])

# Group by PT_ID and count the number of visits for each PT_ID
visit_counts = df.groupby('PT_ID')['ENCOUNTER_DTTM'].count()

# Calculate min, max, and average count
min_count = visit_counts.min()
max_count = visit_counts.max()
average_count = visit_counts.mean()


# Calculate the number of unique patients
num_patients = df['PT_ID'].nunique()

# Randomly select 2000 unique patients
selected_patients = df['PT_ID'].drop_duplicates().sample(n=8000, random_state=1)

# Filter the DataFrame to include only the selected patients
Sample = df[df['PT_ID'].isin(selected_patients)]

# Sort Sample DataFrame by PT_ID and DATE
Sample = Sample.sort_values(by=['PT_ID', 'DATE'])

# Group by PT_ID and count the number of visits for each PT_ID in Sample
visit_counts = Sample.groupby('PT_ID')['ENCOUNTER_DTTM'].count()

# Calculate min, max, and average count in Sample
min_count = visit_counts.min()
max_count = visit_counts.max()
average_count = visit_counts.mean()
max_count_pt_id = visit_counts.idxmax()
max_count_patient_data = Sample[Sample['PT_ID'] == max_count_pt_id]

# Identify no-show patients
no_show_patients = Sample[Sample['STATUS_CD'] == 'No Show']['PT_ID'].unique()
num_ns = len(no_show_patients)

# Identify left without seen patients
Left_without_seen_patients = Sample[Sample['STATUS_CD'] == 'Left without seen']['PT_ID'].unique()
num_lws = len(Left_without_seen_patients)

# Identify canceled patients
Canceled_patients = Sample[Sample['STATUS_CD'] == 'Canceled']['PT_ID'].unique()
num_c = len(Canceled_patients)

# Identify patients with no-shows and cancellations
noshows_and_cancellation_statuses = ['Canceled', 'No Show', 'Left without seen']
noshows_and_cancellation_patients = Sample[Sample['STATUS_CD'].isin(noshows_and_cancellation_statuses)]['PT_ID'].unique()
num_noshows_and_cancellation_patients = len(noshows_and_cancellation_patients)

# Ensure the datetime columns are in the correct format
Sample['ENCOUNTER_DTTM'] = pd.to_datetime(Sample['ENCOUNTER_DTTM'], errors='coerce')
Sample['CNCL_DTTM'] = pd.to_datetime(Sample['CNCL_DTTM'], errors='coerce')

# Calculate time difference in hours between ENCOUNTER_DTTM and CNCL_DTTM
Sample['time_difference_hours'] = (Sample['ENCOUNTER_DTTM'] - Sample['CNCL_DTTM']).dt.total_seconds() / 3600

# Create a new column to indicate if the cancellation was within 24 hours
Sample['cancelled_within_24h'] = Sample['time_difference_hours'].apply(lambda x: 1 if 0 <= x <= 24 else 0)

# -----------------------------------------------------------------------------
# Part 2: Generating Variables, Standardizing Numerical, and Encoding Categorical
# -----------------------------------------------------------------------------

# Ensure datetime columns are in datetime format
datetime_columns = ['ENCOUNTER_DTTM', 'ARRIVED_DTTM', 'CNCL_DTTM', 'RESCHD_DTTM', 'ENCOUNTER_DT', 'SCHD_DTTM']
for col in datetime_columns:
    Sample[col] = pd.to_datetime(Sample[col], format='mixed', errors='coerce', dayfirst=True)

# Generate Rescheduling Interval
#Sample['Rescheduling Interval'] = (Sample['RESCHD_DTTM'] - Sample['SCHD_DTTM']).dt.total_seconds() / 3600

# Generate Cancellation Interval
Sample['Cancellation Interval'] = (Sample['CNCL_DTTM'] - Sample['SCHD_DTTM']).dt.total_seconds() / 3600

# Generate Arrival Interval
#Sample['Arrival Interval'] = (Sample['ARRIVED_DTTM'] - Sample['ENCOUNTER_DTTM']).dt.total_seconds() / 3600

# Capture Cancellation Reasons
Sample['Cancellation Reasons'] = Sample['CNCL_REASON_DESCR']

# Previous Visit Information
Sample = Sample.sort_values(by=['PT_ID', 'ENCOUNTER_DTTM'])

# Previous Visit Count
Sample['Previous Visit Count'] = Sample.groupby('PT_ID').cumcount()

# Time Since Last Visit
Sample['Time Since Last Visit'] = Sample.groupby('PT_ID')['ENCOUNTER_DTTM'].diff().dt.total_seconds() / 3600

# Calculate cumulative counts for different statuses
Sample = Sample.sort_values(by=['PT_ID', 'SCHD_DTTM'])

# Cancellation Count
Sample['Cancellation Count'] = Sample.groupby('PT_ID').apply(lambda x: (x['STATUS_CD'] == 'Canceled').cumsum()).reset_index(level=0, drop=True)

# No show Count
Sample['No Show Count'] = Sample.groupby('PT_ID').apply(lambda x: (x['STATUS_CD'] == 'No Show').cumsum()).reset_index(level=0, drop=True)

# Left without Seen Count
Sample['Left without Seen Count'] = Sample.groupby('PT_ID').apply(lambda x: (x['STATUS_CD'] == 'Left without seen').cumsum()).reset_index(level=0, drop=True)

# Sum of No show and Left without Seen
Sample['No Show + Left without Seen Count'] = Sample['No Show Count'] + Sample['Left without Seen Count']

# Sum of Cancelled, No show, and Left without Seen
Sample['Cancelled + No Show + Left without Seen Count'] = Sample['Cancellation Count'] + Sample['No Show Count'] + Sample['Left without Seen Count']

# Calculate the cumulative count of rescheduled appointments up to each scheduled time
Sample['Reschedule Count'] = Sample.groupby('PT_ID').apply(lambda x: x['RESCHD_DTTM'].notnull().cumsum()).reset_index(level=0, drop=True)

# Total Visits with Same Provider
Sample['Total Visits with Same Provider'] = Sample.groupby(['PT_ID', 'ENCOUNTER_PROV_ID']).cumcount() + 1

# Total Visits in Same Department
Sample['Total Visits in Same Department'] = Sample.groupby(['PT_ID', 'CLIN_DEPT_ABBREV']).cumcount() + 1

# Duration
# Calculate Last Visit Duration
Sample['Last Visit Duration'] = Sample.groupby('PT_ID')['STD_DURATION'].shift(1)

# Calculate Mean Duration Up to the Current Visit
Sample['Mean Duration Up to Current Visit'] = Sample.groupby('PT_ID')['STD_DURATION'].expanding().mean().reset_index(level=0, drop=True)

# Wait Time
#Sample['Wait Time'] = (Sample['ARRIVED_DTTM'] - Sample['ENCOUNTER_DTTM']).dt.total_seconds() / 3600

# Define a function to calculate encounter frequency up to the current scheduled time. Average time between encounters in hours
def calculate_encounter_frequency(group):
    # Calculate the time difference between encounters in hours
    time_diffs = group['ENCOUNTER_DTTM'].diff().dt.total_seconds() / 3600  # in hours  
    # Calculate the cumulative sum of time differences (excluding the first NaN difference)
    cumulative_sum = time_diffs.cumsum()  
    # Calculate the number of intervals (encounter_count - 1)
    interval_count = np.arange(0, len(group))   
    # Calculate the encounter frequency (average time between encounters)
    encounter_frequency = cumulative_sum / interval_count   
    # Handle NaN values for the first encounter
    encounter_frequency.iloc[0] = 0    
    return encounter_frequency

# average time between encounters in hours 
Sample['Encounter Frequency'] = Sample.groupby('PT_ID').apply(calculate_encounter_frequency).reset_index(level=0, drop=True)

# Provider Change Indicator
Sample['Provider Change Indicator'] = Sample.groupby('PT_ID')['ENCOUNTER_PROV_ID'].transform(lambda x: x != x.shift(1))
Sample['Provider Change Count'] = Sample.groupby('PT_ID')['Provider Change Indicator'].cumsum()
Sample['Provider Changed in Last Encounter'] = Sample.groupby('PT_ID')['Provider Change Indicator'].shift(-1).fillna(False)

# Department Change Indicator
Sample['Department Change Indicator'] = Sample.groupby('PT_ID')['CLIN_DEPT_ABBREV'].transform(lambda x: x != x.shift(1))
Sample['Department Change Count'] = Sample.groupby('PT_ID')['Department Change Indicator'].cumsum()
Sample['Department Changed in Last Encounter'] = Sample.groupby('PT_ID')['Department Change Indicator'].shift(-1).fillna(False)

# Arrival Lag
Sample['Arrival Lag'] = (Sample['ARRIVED_DTTM'] - Sample['ENCOUNTER_DTTM']).dt.total_seconds() / 3600
Sample['Last Visit Arrival Lag'] = Sample.groupby('PT_ID')['Arrival Lag'].shift(1)

# Define a function to calculate the average arrival lag up to the current scheduled time
def calculate_average_arrival_lag(group):
    cumulative_sum = group['Arrival Lag'].cumsum().shift()
    encounter_count = np.arange(1, len(group) + 1)
    average_lag = cumulative_sum / encounter_count
    return average_lag.fillna(0)

# Apply the function to each patient group
Sample['Average Arrival Lag Up to Scheduled Time'] = Sample.groupby('PT_ID').apply(calculate_average_arrival_lag).reset_index(level=0, drop=True)

# First/Follow-Up
Sample['First/Follow-Up'] = Sample.groupby('PT_ID').cumcount().apply(lambda x: 'First' if x == 0 else 'Follow-Up')

# Month
Sample['Month'] = Sample['ENCOUNTER_DTTM'].dt.month

# Weekday
Sample['Weekday'] = Sample['ENCOUNTER_DTTM'].dt.weekday

# Add Weekend Indicator Column
Sample['Weekend Indicator'] = Sample['Weekday'].isin([5, 6])

# Extract hour from Visit Time and drop the original Visit Time column
Sample['Visit Time'] = Sample['ENCOUNTER_DTTM'].dt.time
Sample['Visit Hour'] = Sample['Visit Time'].apply(lambda x: x.hour + x.minute/60 + x.second/3600)
Sample = Sample.drop(columns=['Visit Time'])
def categorize_visit_hour(hour):
    if hour < 12:
        return "before 12pm"
    elif 12 <= hour <= 18:
        return "between12-18pm"
    else:
        return "after18pm"

Sample['Visit Hour Category'] = Sample['Visit Hour'].apply(categorize_visit_hour)

# Drop the original Visit Hour column as it's no longer needed
Sample = Sample.drop(columns=['Visit Hour'])

# If the patient did not show up in the last appointment. Previous No-Show (assuming STATUS_CD indicates no-shows)
Sample['Previous No-Show'] = Sample.groupby('PT_ID')['STATUS_CD'].transform(lambda x: x.shift(1).isin(['No Show', 'Left without seen']))

# If the patient canceled the last appointment. Previous Cancel
Sample['Previous Cancel'] = Sample.groupby('PT_ID')['STATUS_CD'].transform(lambda x: x.shift(1) == 'Canceled')

# Add Previous No-Show, Cancel, or Left Without Seen
Sample['Previous No-Show or Cancel or Left Without Seen'] = Sample.groupby('PT_ID')['STATUS_CD'].transform(lambda x: x.shift(1).isin(['No Show', 'Left without seen', 'Canceled']))

# Last Visit Status
Sample['Last Visit Status'] = Sample.groupby('PT_ID')['STATUS_CD'].shift(1)

# Columns to remove
columns_to_remove = [
    'ENCOUNTER_DT', 'ENCOUNTER_ID', 'ARRIVED_DTTM',
    'VISIT_TYPE', 'ENCOUNTER_TYPE', 'ENCOUNTER_PROV_ID', 'ATTNDG_PROV_ID',
    'APPT_PROV_ID', 'CLIN_DEPT_ABBREV', 'CLIN_DEPT_NM', 'FLOOR_LOCATION_NM',
    'STD_DURATION', 'CNCL_DTTM', 'RESCHD_DTTM',
    'CNCL_REASON_DESCR', 'ARRIVED_BY_INITIALS', 'PROV_INTERP_IND'
]

# Remove the specified columns
Sample = Sample.drop(columns=columns_to_remove)

# Save the Sample DataFrame to a CSV file
Sample.to_csv(r"C:/Users/mraja/OneDrive/Desktop/Master Thesis codes/Sample_oversampling_2000.csv")
Sample.to_csv(r"C:/Users/mraja/OneDrive/Desktop/Master Thesis codes/Sample_oversampling_3000.csv")
Sample.to_csv(r"C:/Users/mraja/OneDrive/Desktop/Master Thesis codes/Sample_oversampling_4000.csv")
Sample.to_csv(r"C:/Users/mraja/OneDrive/Desktop/Master Thesis codes/Sample_oversampling_6000.csv")
Sample.to_csv(r"C:/Users/mraja/OneDrive/Desktop/Master Thesis codes/Sample_oversampling_8000.csv")


# Load the Sample DataFrame from the CSV file
Sample = pd.read_csv(r"C:/Users/mraja/OneDrive/Desktop/Master Thesis codes/Sample_oversampling_2000.csv")
Sample = pd.read_csv(r"C:/Users/mraja/OneDrive/Desktop/Master Thesis codes/Sample_oversampling_3000.csv")
Sample = pd.read_csv(r"C:/Users/mraja/OneDrive/Desktop/Master Thesis codes/Sample_oversampling_4000.csv")
Sample = pd.read_csv(r"C:/Users/mraja/OneDrive/Desktop/Master Thesis codes/Sample_oversampling_6000.csv")
Sample = pd.read_csv(r"C:/Users/mraja/OneDrive/Desktop/Master Thesis codes/Sample_oversampling_8000.csv")

# Create a copy of the Sample DataFrame
Sample1 = Sample.copy()
Sample = Sample1.copy()

# List of numerical columns to be standardized
numerical_cols = [
    'Cancellation Interval', 'Previous Visit Count', 'Time Since Last Visit', 'Cancellation Count',
    'No Show Count', 'Left without Seen Count', 'No Show + Left without Seen Count',
    'Cancelled + No Show + Left without Seen Count', 'Reschedule Count',
    'Total Visits with Same Provider', 'Total Visits in Same Department',
    'Last Visit Duration', 'Mean Duration Up to Current Visit',
    'Encounter Frequency', 'Provider Change Count', 'Department Change Count',
    'Last Visit Arrival Lag','Average Arrival Lag Up to Scheduled Time',
]

# Standardize numerical columns
scaler = StandardScaler()
Sample[numerical_cols] = scaler.fit_transform(Sample[numerical_cols])



# Convert Indicator columns to binary
indicator_cols = [
    'Weekend Indicator', 'Previous No-Show', 'Previous Cancel', 
    'Previous No-Show or Cancel or Left Without Seen', 
    'Department Changed in Last Encounter', 'Provider Changed in Last Encounter'
]
Sample[indicator_cols] = Sample[indicator_cols].astype(int)

# One-Hot Encode categorical columns
categorical_cols = [
    'First/Follow-Up', 'Month', 'Weekday', 'Last Visit Status','Visit Hour Category'
]

# Apply one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
encoded_cats = encoder.fit_transform(Sample[categorical_cols])
encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
encoded_cat_df.index = Sample.index

# Concatenate the encoded categorical columns with the Sample DataFrame
Sample = pd.concat([Sample.drop(columns=categorical_cols), encoded_cat_df], axis=1)
print(encoded_cat_df.columns.tolist())

# Create new columns for status conditions
Sample['No Shows'] = Sample['STATUS_CD'].apply(lambda x: 1 if x == 'No Show' else 0)
Sample['No Shows + Left without seen'] = Sample['STATUS_CD'].apply(lambda x: 1 if x in ['No Show', 'Left without seen'] else 0)
Sample['No Shows + Left without seen + Canceled'] = Sample['STATUS_CD'].apply(lambda x: 1 if x in ['No Show', 'Left without seen', 'Canceled'] else 0)
Sample['Canceled'] = Sample['STATUS_CD'].apply(lambda x: 1 if x == 'Canceled' else 0)
Sample['Completed'] = Sample['STATUS_CD'].apply(lambda x: 1 if x == 'Completed' else 0)
Sample['No Shows + Left without seen + cancelled_within_24h'] = Sample['No Shows + Left without seen'] | Sample['cancelled_within_24h']

# Count values of different status columns
no_shows_count = Sample['No Shows'].value_counts()
all_count = Sample['No Shows + Left without seen + Canceled'].value_counts()
all_count1 = Sample['No Shows + Left without seen + cancelled_within_24h'].value_counts()

# Define feature columns
feature_columns = [
    'No Shows', 'No Shows + Left without seen', 'No Shows + Left without seen + Canceled',
    'Canceled', 'Completed', 'No Shows + Left without seen + cancelled_within_24h',
    'Cancellation Interval', 
    'Previous Visit Count', 'Time Since Last Visit',
    'Cancellation Count', 'No Show Count', 'Left without Seen Count',
    'No Show + Left without Seen Count', 'Cancelled + No Show + Left without Seen Count',
    'Reschedule Count', 'Total Visits with Same Provider', 'Total Visits in Same Department',
    'Last Visit Duration', 'Mean Duration Up to Current Visit', 'Visit Time',
    'Encounter Frequency', 'Provider Change Count', 'Provider Changed in Last Encounter',
    'Department Change Count', 'Department Changed in Last Encounter', 'Last Visit Arrival Lag',
    'Average Arrival Lag Up to Scheduled Time', 'Weekend Indicator', 'Previous No-Show',
    'Previous Cancel', 'Previous No-Show or Cancel or Left Without Seen',
    'Cancellation Reasons_Cancelled via Interface', 'Cancellation Reasons_Cancelled via automated reminder system',
    'Cancellation Reasons_Changed by Radiology', 'Cancellation Reasons_Clinically Caused',
    'Cancellation Reasons_Deceased', 'Cancellation Reasons_Deleted via Interface',
    'Cancellation Reasons_Discharged', 'Cancellation Reasons_Displaced Appointment',
    'Cancellation Reasons_Error', 'Cancellation Reasons_Financial',
    'Cancellation Reasons_Financial Concerns', 'Cancellation Reasons_Hospitalized',
    'Cancellation Reasons_Institution', 'Cancellation Reasons_Institution - Appt Made in Error',
    'Cancellation Reasons_Institution - Condition Warrants Cancellation',
    'Cancellation Reasons_Labs Out of Acceptable Range', 'Cancellation Reasons_Lack of Transportation',
    'Cancellation Reasons_Level of Care Change', 'Cancellation Reasons_Moved',
    'Cancellation Reasons_Oncology Treatment Plan Changes', 'Cancellation Reasons_Patient',
    'Cancellation Reasons_Patient - Personal', 'Cancellation Reasons_Patient - Sought Care Elsewhere',
    'Cancellation Reasons_Patient - Weather', 'Cancellation Reasons_Patient Dismissed From Practice',
    'Cancellation Reasons_Personal Reasons', 'Cancellation Reasons_Provider',
    'Cancellation Reasons_Provider - Personal', 'Cancellation Reasons_Provider - Professional',
    'Cancellation Reasons_Provider Departure', 'Cancellation Reasons_Schedule Order Error',
    'Cancellation Reasons_Sought Care Elsewhere', 'Cancellation Reasons_Weather',
    'Cancellation Reasons_nan', 'First/Follow-Up_First', 'First/Follow-Up_Follow-Up',
    'Month_1.0', 'Month_2.0', 'Month_3.0', 'Month_4.0', 'Month_5.0', 'Month_6.0',
    'Month_7.0', 'Month_8.0', 'Month_9.0', 'Month_10.0', 'Month_11.0', 'Month_12.0',
    'Month_nan', 'Weekday_0.0', 'Weekday_1.0', 'Weekday_2.0', 'Weekday_3.0', 'Weekday_4.0',
    'Weekday_5.0', 'Weekday_6.0', 'Weekday_nan', 'Last Visit Status_Arrived',
    'Last Visit Status_Canceled', 'Last Visit Status_Completed', 'Last Visit Status_Left without seen',
    'Last Visit Status_No Show', 'Last Visit Status_nan','Visit Hour Category_, after18pm', 
    'Visit Hour Category_before 12pm', 'Visit Hour Category_ between12-18pm'
]

# Check which columns in 'feature_columns' are actually present in the DataFrame 'Sample'
valid_columns = [col for col in feature_columns if col in Sample.columns]

# Count NaN values before filling
nan_count_before = Sample[valid_columns].isna().sum().sum()

# Fill NaN values with 0
Sample[valid_columns] = Sample[valid_columns].fillna(0)

# Count NaN values after filling
nan_count_after = Sample[valid_columns].isna().sum().sum()

# Prepare the feature matrix 'X' by excluding certain columns
exclude_columns = [
    'No Shows', 'No Shows + Left without seen', 'No Shows + Left without seen + Canceled',
    'Canceled', 'Completed', 'No Shows + Left without seen + cancelled_within_24h'
]
x_columns = [col for col in feature_columns if col in Sample.columns and col not in exclude_columns]
y_column = 'No Shows + Left without seen + cancelled_within_24h'

num_variables = len(x_columns)
print(num_variables)


#Filter out patinet with long sequence

# Group the data by PT_ID and count unique ENCOUNTER_DTTM to get sequence lengths
grouped_by_patient = Sample.groupby('PT_ID')['ENCOUNTER_DTTM'].nunique()

plt.figure(figsize=(10, 6))
plt.hist(grouped_by_patient, bins=50, edgecolor='k', alpha=0.7)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.title('Distribution of Sequence Lengths')
plt.show()

# Determine the threshold for filtering based on the distribution (e.g., 95th percentile)
threshold = grouped_by_patient.quantile(.95)
print("95th percentile threshold for sequence length:", threshold)

# Filter out sequences longer than the threshold
filtered_Sample = Sample[Sample.groupby('PT_ID')['ENCOUNTER_DTTM'].transform('nunique') <= threshold]

print("Original data shape:", Sample.shape)
print("Filtered data shape:", filtered_Sample.shape)



# -----------------------------------------------------------------------------
# Part 3: Data Analysis and Keeping More Effective Variables
# -----------------------------------------------------------------------------
'''
# Exploratory Data Analysis (EDA)
X_EDA = filtered_Sample[x_columns]
Y_EDA = filtered_Sample[[y_column]]

# Plot histograms of the features
X_EDA.hist(bins=30, figsize=(20, 15))
plt.show()

# Define different feature categories for EDA
medical_history = [
    'Previous Visit Count', 'First/Follow-Up_First', 'First/Follow-Up_Follow-Up'
]

appointment_details = [
    'Total Visits with Same Provider', 'Total Visits in Same Department',
    'Provider Change Count', 'Department Change Count', 'Provider Changed in Last Encounter',
    'Department Changed in Last Encounter', 'Visit Hour Category_after18pm', 
    'Visit Hour Category_before 12pm', 'Visit Hour Category_between12-18pm' , 'Last Visit Duration',
    'Mean Duration Up to Current Visit', 'Encounter Frequency', 'Month_1.0', 'Month_2.0',
    'Month_3.0', 'Month_4.0', 'Month_5.0', 'Month_6.0', 'Month_7.0', 'Month_8.0',
    'Month_9.0', 'Month_10.0', 'Month_11.0', 'Month_12.0', 'Month_nan', 'Weekday_0.0',
    'Weekday_1.0', 'Weekday_2.0', 'Weekday_3.0', 'Weekday_4.0', 'Weekday_5.0',
    'Weekday_6.0', 'Weekday_nan'
]

patient_behavior = [
    'Cancellation Count', 'No Show Count', 'Left without Seen Count',
    'No Show + Left without Seen Count', 'Cancelled + No Show + Left without Seen Count',
    'Reschedule Count', 'Previous No-Show', 'Previous Cancel', 'Previous No-Show or Cancel or Left Without Seen',
    'Last Visit Status_Canceled', 'Last Visit Status_Completed', 'Last Visit Status_Left without seen',
    'Last Visit Status_No Show', 'Last Visit Status_nan'
]

temporal_variables = [
    'Time Since Last Visit', 'Weekend Indicator', 'Cancellation Interval',
    'Last Visit Arrival Lag','Average Arrival Lag Up to Scheduled Time'
]

cancellation_reasons = [
    'Cancellation Reasons_Cancelled via Interface', 'Cancellation Reasons_Cancelled via automated reminder system',
    'Cancellation Reasons_Clinically Caused', 'Cancellation Reasons_Deceased', 'Cancellation Reasons_Deleted via Interface',
    'Cancellation Reasons_Displaced Appointment', 'Cancellation Reasons_Error', 'Cancellation Reasons_Financial Concerns',
    'Cancellation Reasons_Hospitalized', 'Cancellation Reasons_Institution', 'Cancellation Reasons_Institution - Appt Made in Error',
    'Cancellation Reasons_Institution - Condition Warrants Cancellation', 'Cancellation Reasons_Lack of Transportation',
    'Cancellation Reasons_Moved', 'Cancellation Reasons_Oncology Treatment Plan Changes', 'Cancellation Reasons_Patient',
    'Cancellation Reasons_Patient - Personal', 'Cancellation Reasons_Patient - Weather', 'Cancellation Reasons_Patient Dismissed From Practice',
    'Cancellation Reasons_Personal Reasons', 'Cancellation Reasons_Provider', 'Cancellation Reasons_Provider - Personal',
    'Cancellation Reasons_Provider - Professional', 'Cancellation Reasons_Provider Departure', 'Cancellation Reasons_Schedule Order Error',
    'Cancellation Reasons_Sought Care Elsewhere', 'Cancellation Reasons_nan'
]

# Function to calculate the correlation for each category
def calculate_category_correlation(features, df, target):
    correlations = df[features].apply(lambda x: x.corr(df[target]))
    return correlations.abs().mean()  # Return the mean absolute correlation

# Calculate correlations for each category
medical_history_corr = calculate_category_correlation(medical_history, filtered_Sample, y_column)
appointment_details_corr = calculate_category_correlation(appointment_details, filtered_Sample, y_column)
patient_behavior_corr = calculate_category_correlation(patient_behavior, filtered_Sample, y_column)
temporal_variables_corr = calculate_category_correlation(temporal_variables, filtered_Sample, y_column)
cancellation_reasons_corr = calculate_category_correlation(cancellation_reasons, filtered_Sample, y_column)

# Create a DataFrame to compare correlations
correlation_comparison = pd.DataFrame({
    'Category': ['Medical History', 'Appointment Details', 'Patient Behavior', 'Temporal Variables', 'Cancellation Reasons'],
    'Mean Absolute Correlation': [medical_history_corr, appointment_details_corr, patient_behavior_corr, temporal_variables_corr, cancellation_reasons_corr]
})

# Sort by correlation
correlation_comparison = correlation_comparison.sort_values(by='Mean Absolute Correlation', ascending=False)

# Plot the correlation comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Mean Absolute Correlation', y='Category', data=correlation_comparison, palette='viridis')
plt.title('Mean Absolute Correlation of Categories with Target Variable')
plt.xlabel('Mean Absolute Correlation')
plt.ylabel('Category')
plt.show()

# Calculate and plot correlations for individual variables within each category
individual_correlations = {}

# Calculate correlations for individual variables in each category
for category, features in {
    'Medical History': medical_history,
    'Appointment Details': appointment_details,
    'Patient Behavior': patient_behavior,
    'Temporal Variables': temporal_variables,
    'Cancellation Reasons': cancellation_reasons
}.items():
    individual_correlations[category] = filtered_Sample[features].apply(lambda x: x.corr(filtered_Sample[y_column])).abs()

# Plot individual correlations
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 25))
fig.subplots_adjust(hspace=0.5)

for ax, (category, correlations) in zip(axes, individual_correlations.items()):
    correlations.sort_values(ascending=False).plot(kind='bar', ax=ax, title=f'Correlations of {category} Variables with Target Variable')

plt.show()

# Identify pairs of highly correlated features (correlation > 0.8 or < -0.8)
threshold = 0.8
highly_correlated_pairs = [
    (var1, var2) for var1 in X_EDA.columns 
    for var2 in X_EDA.columns 
    if var1 != var2 and abs(X_EDA[var1].corr(X_EDA[var2])) > threshold
]

# Display highly correlated pairs
print("Highly Correlated Pairs (|correlation| > 0.8):")
for var1, var2 in highly_correlated_pairs:
    print(f"{var1} and {var2}: {X_EDA[var1].corr(X_EDA[var2]):.2f}")

# Concatenate X and Y for correlation calculation
data_EDA = pd.concat([X_EDA, Y_EDA], axis=1)

# Calculate correlation matrix
corr_matrix = data_EDA.corr()

# Extract correlation with target variable
target_corr = corr_matrix[Y_EDA.columns[0]].drop(Y_EDA.columns[0])  # Drop target itself

# Display the correlations
print("Correlation with target variable:")
print(target_corr.sort_values(ascending=False))

# Plot correlation of features with the target variable
plt.figure(figsize=(12, 8))
sns.barplot(x=target_corr.index, y=target_corr.values, palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Correlation of Features with Target Variable')
plt.show()

# Determine which feature to keep in each pair of highly correlated features
features_to_keep = set(target_corr.index)  # Start with all features
for var1, var2 in highly_correlated_pairs:
    if abs(target_corr[var1]) > abs(target_corr[var2]):
        features_to_keep.discard(var2)
    else:
        features_to_keep.discard(var1)

# Create a new DataFrame with only the selected features
X_selected = X_EDA[list(features_to_keep)]
num_variables = X_selected.shape[1]
print(f"The number of variables in X_selected is: {num_variables}")

# Get the list of selected feature columns
X_selected_columns = X_selected.columns.tolist()
'''
# -----------------------------------------------------------------------------
# Part 4: Splitting Data, Oversampling, and Padding
# -----------------------------------------------------------------------------

# Assuming X_train_sample_flat is the flattened version of your training data
feature_names = [
    'Previous Visit Count', 'First/Follow-Up_First', 'First/Follow-Up_Follow-Up',
    'Total Visits with Same Provider', 'Total Visits in Same Department',
    'Provider Change Count', 'Department Change Count', 'Provider Changed in Last Encounter',
    'Department Changed in Last Encounter', 'Visit Hour Category_after18pm',
    'Visit Hour Category_before 12pm', 'Visit Hour Category_between12-18pm', 'Last Visit Duration',
    'Mean Duration Up to Current Visit', 'Encounter Frequency', 'Month_1.0', 'Month_2.0',
    'Month_3.0', 'Month_4.0', 'Month_5.0', 'Month_6.0', 'Month_7.0', 'Month_8.0',
    'Month_9.0', 'Month_10.0', 'Month_11.0', 'Month_12.0', 'Month_nan', 'Weekday_0.0',
    'Weekday_1.0', 'Weekday_2.0', 'Weekday_3.0', 'Weekday_4.0', 'Weekday_5.0',
    'Weekday_6.0', 'Weekday_nan', 'Cancellation Count', 'No Show Count', 'Left without Seen Count',
    'No Show + Left without Seen Count', 'Cancelled + No Show + Left without Seen Count',
    'Reschedule Count', 'Previous No-Show', 'Previous Cancel', 'Previous No-Show or Cancel or Left Without Seen',
    'Last Visit Status_Canceled', 'Last Visit Status_Completed', 'Last Visit Status_Left without seen',
    'Last Visit Status_No Show', 'Last Visit Status_nan', 'Time Since Last Visit', 'Weekend Indicator',
    'Cancellation Interval', 'Last Visit Arrival Lag', 'Average Arrival Lag Up to Scheduled Time',
    'Cancellation Reasons_Cancelled via Interface', 'Cancellation Reasons_Cancelled via automated reminder system',
    'Cancellation Reasons_Clinically Caused', 'Cancellation Reasons_Deceased', 'Cancellation Reasons_Deleted via Interface',
    'Cancellation Reasons_Displaced Appointment', 'Cancellation Reasons_Error', 'Cancellation Reasons_Financial Concerns',
    'Cancellation Reasons_Hospitalized', 'Cancellation Reasons_Institution', 'Cancellation Reasons_Institution - Appt Made in Error',
    'Cancellation Reasons_Institution - Condition Warrants Cancellation', 'Cancellation Reasons_Lack of Transportation',
    'Cancellation Reasons_Moved', 'Cancellation Reasons_Oncology Treatment Plan Changes', 'Cancellation Reasons_Patient',
    'Cancellation Reasons_Patient - Personal', 'Cancellation Reasons_Patient - Weather', 'Cancellation Reasons_Patient Dismissed From Practice',
    'Cancellation Reasons_Personal Reasons', 'Cancellation Reasons_Provider', 'Cancellation Reasons_Provider - Personal',
    'Cancellation Reasons_Provider - Professional', 'Cancellation Reasons_Provider Departure', 'Cancellation Reasons_Schedule Order Error',
    'Cancellation Reasons_Sought Care Elsewhere', 'Cancellation Reasons_nan'
]

# Feature categories mapping as given
feature_categories = {
    'medical_history': [
        'Previous Visit Count', 'First/Follow-Up_First', 'First/Follow-Up_Follow-Up'
    ],
    'appointment_details': [
        'Total Visits with Same Provider', 'Total Visits in Same Department',
        'Provider Change Count', 'Department Change Count', 'Provider Changed in Last Encounter',
        'Department Changed in Last Encounter', 'Visit Hour Category_after18pm',
        'Visit Hour Category_before 12pm', 'Visit Hour Category_between12-18pm', 'Last Visit Duration',
        'Mean Duration Up to Current Visit', 'Encounter Frequency', 'Month_1.0', 'Month_2.0',
        'Month_3.0', 'Month_4.0', 'Month_5.0', 'Month_6.0', 'Month_7.0', 'Month_8.0',
        'Month_9.0', 'Month_10.0', 'Month_11.0', 'Month_12.0', 'Month_nan', 'Weekday_0.0',
        'Weekday_1.0', 'Weekday_2.0', 'Weekday_3.0', 'Weekday_4.0', 'Weekday_5.0',
        'Weekday_6.0', 'Weekday_nan'
    ],
    'patient_behavior': [
        'Cancellation Count', 'No Show Count', 'Left without Seen Count',
        'No Show + Left without Seen Count', 'Cancelled + No Show + Left without Seen Count',
        'Reschedule Count', 'Previous No-Show', 'Previous Cancel', 'Previous No-Show or Cancel or Left Without Seen',
        'Last Visit Status_Canceled', 'Last Visit Status_Completed', 'Last Visit Status_Left without seen',
        'Last Visit Status_No Show', 'Last Visit Status_nan'
    ],
    'temporal_variables': [
        'Time Since Last Visit', 'Weekend Indicator', 'Cancellation Interval',
        'Last Visit Arrival Lag', 'Average Arrival Lag Up to Scheduled Time'
    ]
}


'''
    'cancellation_reasons': [
        'Cancellation Reasons_Cancelled via Interface', 'Cancellation Reasons_Cancelled via automated reminder system',
        'Cancellation Reasons_Clinically Caused', 'Cancellation Reasons_Deceased', 'Cancellation Reasons_Deleted via Interface',
        'Cancellation Reasons_Displaced Appointment', 'Cancellation Reasons_Error', 'Cancellation Reasons_Financial Concerns',
        'Cancellation Reasons_Hospitalized', 'Cancellation Reasons_Institution', 'Cancellation Reasons_Institution - Appt Made in Error',
        'Cancellation Reasons_Institution - Condition Warrants Cancellation', 'Cancellation Reasons_Lack of Transportation',
        'Cancellation Reasons_Moved', 'Cancellation Reasons_Oncology Treatment Plan Changes', 'Cancellation Reasons_Patient',
        'Cancellation Reasons_Patient - Personal', 'Cancellation Reasons_Patient - Weather', 'Cancellation Reasons_Patient Dismissed From Practice',
        'Cancellation Reasons_Personal Reasons', 'Cancellation Reasons_Provider', 'Cancellation Reasons_Provider - Personal',
        'Cancellation Reasons_Provider - Professional', 'Cancellation Reasons_Provider Departure', 'Cancellation Reasons_Schedule Order Error',
        'Cancellation Reasons_Sought Care Elsewhere', 'Cancellation Reasons_nan'
    ]
'''


# Map feature names to indices
feature_indices = {name: idx for idx, name in enumerate(feature_names)}

# Extract feature indices for each category
category_feature_indices = {cat: [feature_indices[feat] for feat in feats] for cat, feats in feature_categories.items()}




X_selected_columns = []
for category, features in feature_categories.items():
    X_selected_columns.extend(features)




# Mapping from cancellation reasons to categories
mapping = {
    'Body Habitus': 'medical_history',
    'Canceled Via Automated Reminder System': 'appointment_details',
    'Cancelled Via Interface': 'appointment_details',
    'Changed By Radiology': 'appointment_details',
    'Claustrophobia': 'medical_history',
    'Clinically Caused': 'medical_history',
    'Deceased': 'medical_history',
    'Deleted Via Interface': 'appointment_details',
    'Discharged': 'medical_history',
    'Displaced Appointment': 'appointment_details',
    'Edu/Meeting': 'patient_behavior',
    'Error': 'appointment_details',
    'Feeling Better': 'medical_history',
    'Financial': 'patient_behavior',
    'Hospitalized': 'medical_history',
    'Implant (Undocumented)': 'medical_history',
    'Improper Iv Access/Infiltrate Iv': 'medical_history',
    'Incompatible Implant': 'medical_history',
    'Institution': 'appointment_details',
    'Institution - Weather': 'temporal_variables',
    'Labs Out Of Acceptable Range': 'medical_history',
    'Lack Of Transportation': 'patient_behavior',
    'Level Of Care Change': 'medical_history',
    'Md Appointment': 'appointment_details',
    'Member Terminated': 'patient_behavior',
    'Moved': 'patient_behavior',
    'No Interpreter Available': 'appointment_details',
    'Oncology Treatment Plan Changes': 'medical_history',
    'Order Discontinued': 'medical_history',
    'Patient': 'patient_behavior',
    'Patient - Personal': 'patient_behavior',
    'Patient - Sought Care Elsewhere': 'patient_behavior',
    'Patient - Weather': 'temporal_variables',
    'Patient Dismissed From Practice': 'patient_behavior',
    'Patient Not Cleared By Pcp For Procedure': 'medical_history',
    'Pregnant': 'medical_history',
    'Prep/Med/Results Unavailable': 'medical_history',
    'Provider': 'appointment_details',
    'Provider - Personal': 'appointment_details',
    'Provider - Professional': 'appointment_details',
    'Provider Departure': 'appointment_details',
    'Room/Resource Maintenance': 'appointment_details',
    'Schedule Order Error': 'appointment_details',
    'Scheduled From Wait List': 'appointment_details',
    'Unhappy/Changed Provider': 'patient_behavior',
    'Weather': 'temporal_variables',
    'Delivered/No Longer Pregnant': 'medical_history',
    'Personal Reasons': 'patient_behavior',
    'Institution - Appt Made In Error': 'appointment_details',
    'Sought Care Elsewhere': 'patient_behavior',
    'Institution - Condition Warrants Cancellation': 'medical_history',
    'Financial Concerns': 'patient_behavior',
    'Cancelled Via Automated Reminder System': 'appointment_details'
}


# Function to generate category-specific labels based on no-show status and cancellation reasons
def generate_category_labels(df, no_show_col, cancellation_reason_col, mapping):
    categories = ['medical_history', 'appointment_details', 'patient_behavior', 'temporal_variables']
    for category in categories:
        df[f'{category}_label'] = 0
    
    for index, row in df.iterrows():
        if row[no_show_col] == 1:
            cancellation_reason = row[cancellation_reason_col]
            if pd.isna(cancellation_reason):
                category = 'patient_behavior'
            else:
                cancellation_reason = cancellation_reason.title()  # Ensure matching with title case
                category = mapping.get(cancellation_reason, 'patient_behavior')
            df.at[index, f'{category}_label'] = 1
    
    return df

# Generate labels in the dataset
filtered_Sample = generate_category_labels(filtered_Sample, 'No Shows + Left without seen + cancelled_within_24h', 'Cancellation Reasons', mapping)

# Now the filtered_Sample dataframe contains the new label columns
print(filtered_Sample)

# Define label columns
y_no_show = filtered_Sample['No Shows + Left without seen + cancelled_within_24h']
y_medical_history = filtered_Sample['medical_history_label']
y_appointment_details = filtered_Sample['appointment_details_label']
y_patient_behavior = filtered_Sample['patient_behavior_label']
y_temporal_variables = filtered_Sample['temporal_variables_label']

# Count the occurrences of 1 for each label
count_no_show = y_no_show.sum()
count_medical_history = y_medical_history.sum()
count_appointment_details = y_appointment_details.sum()
count_patient_behavior = y_patient_behavior.sum()
count_temporal_variables = y_temporal_variables.sum()

print(f"Count of 'No Shows + Left without seen + cancelled_within_24h': {count_no_show}")
print(f"Count of 'medical_history_label': {count_medical_history}")
print(f"Count of 'appointment_details_label': {count_appointment_details}")
print(f"Count of 'patient_behavior_label': {count_patient_behavior}")
print(f"Count of 'temporal_variables_label': {count_temporal_variables}")



# Combine labels into a dictionary
y = {
    'No Shows + Left without seen + cancelled_within_24h': y_no_show,
    'medical_history': y_medical_history,
    'appointment_details': y_appointment_details,
    'patient_behavior': y_patient_behavior,
    'temporal_variables': y_temporal_variables
}

# Split data before oversampling and padding
X = filtered_Sample[X_selected_columns]


PT_ID = 'PT_ID'  # replace with your actual PT_ID column
ENCOUNTER_DTTM = 'ENCOUNTER_DTTM'  # replace with your actual ENCOUNTER_DTTM column

# Ensure all required columns are present
required_columns = X_selected_columns + [y_column] + [f'{key}_label' for key in y.keys() if key != y_column] + [PT_ID, ENCOUNTER_DTTM]
if not all(col in filtered_Sample.columns for col in required_columns):
    missing_cols = [col for col in required_columns if col not in filtered_Sample.columns]
    raise ValueError(f"Missing columns in the DataFrame: {missing_cols}")
    
    
# Create the final DataFrame
final_df = filtered_Sample[required_columns]
print("Shape of final_df:", final_df.shape)

# Group the data by PT_ID
grouped = final_df.groupby(PT_ID)

# Initialize lists to store sequences and labels
X_sequences = []
y_sequences = {key: [] for key in y.keys()}
pt_id_sequences = []
ENCOUNTER_DTTM_sequences = []

# Iterate over each group (grouped by PT_ID)
for name, group in grouped:
    # Sort the group by ENCOUNTER_DTTM
    group = group.sort_values(by=ENCOUNTER_DTTM)
    
    # Extract features
    features = group[X_selected_columns].values
    pt_ids = group[PT_ID].values
    ENCOUNTER_DTTMs = group[ENCOUNTER_DTTM].values
    
    # Extract labels
    labels = {key: group[f'{key}_label'].values for key in y.keys() if key != y_column}
    labels[y_column] = group[y_column].values
    
    # Append the features and labels to the respective lists
    X_sequences.append(features)
    for key in y.keys():
        y_sequences[key].append(labels[key])
    pt_id_sequences.append(pt_ids)
    ENCOUNTER_DTTM_sequences.append(ENCOUNTER_DTTMs)

# Print the number of sequences
print("Number of sequences:", len(X_sequences))

# Flatten the labels for stratification (using 'No Shows + Left without seen + cancelled_within_24h' for stratification)
flat_labels = np.array([seq[-1] for seq in y_sequences['No Shows + Left without seen + cancelled_within_24h']])

# Split data into training + validation and test sets (80% train+val, 20% test)
X_train_val, X_test, pt_id_train_val, pt_id_test, ENCOUNTER_DTTM_train_val, ENCOUNTER_DTTM_test = train_test_split(
    X_sequences, pt_id_sequences, ENCOUNTER_DTTM_sequences, test_size=0.2, random_state=42, stratify=flat_labels)

# Split labels into training + validation and test sets
y_train_val = {key: [] for key in y.keys()}
y_test = {key: [] for key in y.keys()}

for key in y.keys():
    y_train_val[key], y_test[key] = train_test_split(y_sequences[key], test_size=0.2, random_state=42, stratify=flat_labels)

# Extract the last valid labels for the training + validation set for stratification
train_val_labels = np.array([seq[-1] for seq in y_train_val['No Shows + Left without seen + cancelled_within_24h']])

# Split training + validation set into separate training and validation sets (75% train, 25% val of train+val set)
X_train, X_val, pt_id_train, pt_id_val, ENCOUNTER_DTTM_train, ENCOUNTER_DTTM_val = train_test_split(
    X_train_val, pt_id_train_val, ENCOUNTER_DTTM_train_val, test_size=0.25, random_state=42, stratify=train_val_labels)

# Split labels into training and validation sets
y_train = {key: [] for key in y.keys()}
y_val = {key: [] for key in y.keys()}

for key in y.keys():
    y_train[key], y_val[key] = train_test_split(y_train_val[key], test_size=0.25, random_state=42, stratify=train_val_labels)

# Print the sizes of the splits
print("X_train size:", len(X_train))
print("y_train overall no-show size:", len(y_train['No Shows + Left without seen + cancelled_within_24h']))
print("y_train medical history size:", len(y_train['medical_history']))
print("y_train appointment details size:", len(y_train['appointment_details']))
print("y_train patient behavior size:", len(y_train['patient_behavior']))
print("y_train temporal variables size:", len(y_train['temporal_variables']))

print("X_val size:", len(X_val))
print("y_val overall no-show size:", len(y_val['No Shows + Left without seen + cancelled_within_24h']))
print("y_val medical history size:", len(y_val['medical_history']))
print("y_val appointment details size:", len(y_val['appointment_details']))
print("y_val patient behavior size:", len(y_val['patient_behavior']))
print("y_val temporal variables size:", len(y_val['temporal_variables']))





# Initialize lists to store flattened sequences and their corresponding PT_ID and ENCOUNTER_DTTM
X_train_flat = []
y_train_flat = {key: [] for key in y.keys()}
pt_id_flat = []
ENCOUNTER_DTTM_flat = []

# Flatten the training data
for seq, labels, pt_ids, ENCOUNTER_DTTMs in zip(X_train, zip(*[y_train[key] for key in y.keys()]), pt_id_train, ENCOUNTER_DTTM_train):
    X_train_flat.extend(seq)
    for key, label_seq in zip(y.keys(), labels):
        y_train_flat[key].extend(label_seq)
    pt_id_flat.extend(pt_ids)
    ENCOUNTER_DTTM_flat.extend(ENCOUNTER_DTTMs)

# Create DataFrame for the flattened train data
train_df = pd.DataFrame(X_train_flat, columns=X_selected_columns)
for key in y.keys():
    train_df[key] = y_train_flat[key]
train_df[PT_ID] = pt_id_flat
train_df[ENCOUNTER_DTTM] = ENCOUNTER_DTTM_flat

# Ensure PT_ID and ENCOUNTER_DTTM are treated as scalars
train_df[PT_ID] = train_df[PT_ID].astype(str)
train_df[ENCOUNTER_DTTM] = pd.to_datetime(train_df[ENCOUNTER_DTTM])

# Print the shape of the train_df
print("train_df shape:", train_df.shape)
print(train_df.head())

# Group the data by PT_ID and find the maximum sequence length
grouped_by_patient = train_df.groupby(PT_ID)['ENCOUNTER_DTTM'].nunique()
max_seq_length = grouped_by_patient.max()
print("grouped_by_patient shape:", grouped_by_patient.shape)

# Create dictionaries to hold oversampled sequences
X_train_oversampled = []
y_train_oversampled = {key: [] for key in y.keys()}
pt_id_oversampled = []
ENCOUNTER_DTTM_oversampled = []

# Iterate over each timestep
for seq_idx in range(max_seq_length):
    # Extract all timesteps data at index seq_idx for each patient
    current_timesteps_X = []
    current_timesteps_y = {key: [] for key in y.keys()}
    current_timesteps_pt_id = []
    current_timesteps_ENCOUNTER_DTTM = []

    for X_seq, y_seq, pt_id_seq, ENCOUNTER_DTTM_seq in zip(X_train, zip(*[y_train[key] for key in y.keys()]), pt_id_train, ENCOUNTER_DTTM_train):
        if seq_idx < len(X_seq):
            current_timesteps_X.append(X_seq[seq_idx])
            for key, label_seq in zip(y.keys(), y_seq):
                current_timesteps_y[key].append(label_seq[seq_idx])
            current_timesteps_pt_id.append(pt_id_seq)
            current_timesteps_ENCOUNTER_DTTM.append(ENCOUNTER_DTTM_seq[seq_idx])

    # Count the labels at the current timestep for overall no-shows
    label_counts = np.bincount(current_timesteps_y['No Shows + Left without seen + cancelled_within_24h'])
    unique_classes = np.unique(current_timesteps_y['No Shows + Left without seen + cancelled_within_24h'])

    # Print the sizes and class distribution before oversampling
    print(f"Timestep {seq_idx}:")
    print("Size before oversampling:", len(current_timesteps_X))
    print("Class distribution before oversampling:", dict(zip(unique_classes, label_counts)))

    # Determine the minority and majority classes for overall no-shows
    minority_class = np.argmin(label_counts)
    majority_class = np.argmax(label_counts)

    # Calculate how many samples to add for oversampling
    num_to_add = label_counts[majority_class] - label_counts[minority_class]

    # Extract minority class samples
    minority_samples_X = [x for x, y in zip(current_timesteps_X, current_timesteps_y['No Shows + Left without seen + cancelled_within_24h']) if y == minority_class]
    minority_samples_y = {key: [current_timesteps_y[key][i] for i, y in enumerate(current_timesteps_y['No Shows + Left without seen + cancelled_within_24h']) if y == minority_class] for key in y.keys()}
    minority_samples_pt_id = [pt_id for pt_id, y in zip(current_timesteps_pt_id, current_timesteps_y['No Shows + Left without seen + cancelled_within_24h']) if y == minority_class]
    minority_samples_ENCOUNTER_DTTM = [ENCOUNTER_DTTM for ENCOUNTER_DTTM, y in zip(current_timesteps_ENCOUNTER_DTTM, current_timesteps_y['No Shows + Left without seen + cancelled_within_24h']) if y == minority_class]

    # Resample minority class samples to match the majority class
    if num_to_add > 0:
        resampled_X, resampled_y_list, resampled_pt_id, resampled_ENCOUNTER_DTTM = resample(
            minority_samples_X, list(zip(*[minority_samples_y[key] for key in y.keys()])), minority_samples_pt_id, minority_samples_ENCOUNTER_DTTM,
            replace=True, n_samples=num_to_add, random_state=42)

        resampled_y = {key: [y_list[i] for y_list in resampled_y_list] for i, key in enumerate(y.keys())}

        # Add resampled data to current timesteps data
        current_timesteps_X.extend(resampled_X)
        for key in y.keys():
            current_timesteps_y[key].extend(resampled_y[key])
        current_timesteps_pt_id.extend(resampled_pt_id)
        current_timesteps_ENCOUNTER_DTTM.extend(resampled_ENCOUNTER_DTTM)

    # Add the current timestep data to the oversampled sequences
    for i in range(len(current_timesteps_X)):
        if i < len(X_train_oversampled):
            X_train_oversampled[i].append(current_timesteps_X[i])
            for key in y.keys():
                y_train_oversampled[key][i].append(current_timesteps_y[key][i])
            pt_id_oversampled[i].append(current_timesteps_pt_id[i])
            ENCOUNTER_DTTM_oversampled[i].append(current_timesteps_ENCOUNTER_DTTM[i])
        else:
            X_train_oversampled.append([current_timesteps_X[i]])
            for key in y.keys():
                y_train_oversampled[key].append([current_timesteps_y[key][i]])
            pt_id_oversampled.append([current_timesteps_pt_id[i]])
            ENCOUNTER_DTTM_oversampled.append([current_timesteps_ENCOUNTER_DTTM[i]])

    # Count the labels after oversampling
    label_counts_after = {key: np.bincount(current_timesteps_y[key]) for key in y.keys() if len(current_timesteps_y[key]) > 0}
    unique_classes_after = {key: np.unique(current_timesteps_y[key]) for key in y.keys() if len(current_timesteps_y[key]) > 0}

    # Print the sizes and class distribution after oversampling
    print("Size after oversampling:", len(current_timesteps_X))
    for key in y.keys():
        if key in label_counts_after:
            print(f"Category: {key}")
            print("Class distribution after oversampling:", dict(zip(unique_classes_after[key], label_counts_after[key])))
    print("")
    
    
    
# Flatten the oversampled sequences and initialize lists to store the flattened data
X_train_flat_oversampled = []
y_train_flat_oversampled = {key: [] for key in y.keys()}
pt_id_flat_oversampled = []
ENCOUNTER_DTTM_flat_oversampled = []

for seq, labels, pt_ids, ENCOUNTER_DTTMs in zip(X_train_oversampled, zip(*[y_train_oversampled[key] for key in y.keys()]), pt_id_oversampled, ENCOUNTER_DTTM_oversampled):
    X_train_flat_oversampled.extend(seq)
    for key, label_seq in zip(y.keys(), labels):
        y_train_flat_oversampled[key].extend(label_seq)
    pt_id_flat_oversampled.extend(pt_ids)
    ENCOUNTER_DTTM_flat_oversampled.extend(ENCOUNTER_DTTMs)

# Create DataFrame for the oversampled train data
train_df_oversampled = pd.DataFrame(X_train_flat_oversampled, columns=X_selected_columns)
for key in y.keys():
    train_df_oversampled[key] = y_train_flat_oversampled[key]
train_df_oversampled[PT_ID] = pt_id_flat_oversampled
train_df_oversampled[ENCOUNTER_DTTM] = ENCOUNTER_DTTM_flat_oversampled

# Ensure PT_ID and ENCOUNTER_DTTM are treated as scalars in the oversampled DataFrame
train_df_oversampled[PT_ID] = train_df_oversampled[PT_ID].astype(str)
train_df_oversampled[ENCOUNTER_DTTM] = pd.to_datetime(train_df_oversampled[ENCOUNTER_DTTM])

# Print the shape of the oversampled train_df
print("train_df_oversampled shape:", train_df_oversampled.shape)
print(train_df_oversampled.head())



# Group by PT_ID and count unique ENCOUNTER_DTTM values after oversampling
grouped_by_patient_oversampled = train_df_oversampled.groupby('PT_ID')['ENCOUNTER_DTTM'].nunique()
max_seq_length_oversampled = grouped_by_patient_oversampled.max()
print("Maximum sequence length by grouping PT_ID and ENCOUNTER_DTTM after oversampling:", max_seq_length_oversampled)

# Placeholder values for padding
padding_value = 99  # Use a distinct value that does not occur in your dataset

# Function to determine the maximum sequence length
def get_max_seq_length(df, pt_id_col, date_col):
    grouped_by_patient = df.groupby(pt_id_col)[date_col].nunique()
    max_seq_length = grouped_by_patient.max()
    return max_seq_length

# Determine maximum sequence length from final_df
max_seq_length = get_max_seq_length(train_df_oversampled, PT_ID, ENCOUNTER_DTTM)
print("Maximum sequence length:", max_seq_length)

# Function to pad and truncate sequences
def pad_sequences(sequences, labels, max_length, padding_value):
    padded_sequences = []
    padded_labels = {key: [] for key in labels.keys()}

    for seq in sequences:
        seq = np.array(seq)
        num_padding = max_length - len(seq)

        if num_padding > 0:
            padding_data = np.full((num_padding, seq.shape[1]), padding_value)
            seq = np.concatenate([seq, padding_data])
        elif num_padding < 0:
            seq = seq[:max_length]

        padded_sequences.append(seq)

    for key in labels.keys():
        for lbl in labels[key]:
            lbl = np.array(lbl)
            num_padding = max_length - len(lbl)

            if num_padding > 0:
                padding_labels = np.full((num_padding,), padding_value)
                lbl = np.concatenate([lbl, padding_labels])
            elif num_padding < 0:
                lbl = lbl[:max_length]

            padded_labels[key].append(lbl)

    padded_sequences = np.array(padded_sequences)
    for key in padded_labels.keys():
        padded_labels[key] = np.array(padded_labels[key])

    return padded_sequences, padded_labels

# Apply padding to train, validation, and test data
X_train_padded, y_train_padded = pad_sequences(X_train_oversampled, y_train_oversampled, max_seq_length, padding_value)
X_val_padded, y_val_padded = pad_sequences(X_val, y_val, max_seq_length, padding_value)
X_test_padded, y_test_padded = pad_sequences(X_test, y_test, max_seq_length, padding_value)

# Reshape the labels to add an extra dimension
y_train_padded = {key: np.expand_dims(y_train_padded[key], axis=-1) for key in y_train_padded}
y_val_padded = {key: np.expand_dims(y_val_padded[key], axis=-1) for key in y_val_padded}
y_test_padded = {key: np.expand_dims(y_test_padded[key], axis=-1) for key in y_test_padded}

# Verify the reshaped labels
print("Padded Train Data shape:", X_train_padded.shape)
for key in y_train_padded:
    print(f"Padded Train Labels shape for {key}:", y_train_padded[key].shape)
print("Padded Validation Data shape:", X_val_padded.shape)
for key in y_val_padded:
    print(f"Padded Validation Labels shape for {key}:", y_val_padded[key].shape)
print("Padded Test Data shape:", X_test_padded.shape)
for key in y_test_padded:
    print(f"Padded Test Labels shape for {key}:", y_test_padded[key].shape)

# -----------------------------------------------------------------------------
# Part 5: Model Definition and Training
# -----------------------------------------------------------------------------
# Define the number of features
num_features = X_train_padded.shape[2]

# Custom Dense Layer with Variational Inference
class MyDenseVariational(Layer):
    def __init__(self, units, kl_weight=1.0, activation='sigmoid', **kwargs):
        super(MyDenseVariational, self).__init__(**kwargs)
        self.units = units
        self.kl_weight = kl_weight
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # Initialize weights for the posterior distribution
        self.w_mean = self.add_weight(name="w_mean", shape=(input_shape[-1], self.units),
                                      initializer='random_normal', trainable=True)
        self.w_stddev = self.add_weight(name="w_stddev", shape=(input_shape[-1], self.units),
                                        initializer='random_normal', trainable=True)
        self.bias = self.add_weight(name="bias", shape=(self.units,), initializer="zeros", trainable=True)

    def call(self, inputs, mask=None):
        w_posterior = tfp.distributions.Normal(loc=self.w_mean, scale=tf.nn.softplus(self.w_stddev))
        w_sample = w_posterior.sample()
        kl_divergence = tfp.distributions.kl_divergence(w_posterior, tfp.distributions.Normal(loc=0., scale=1.))
        self.add_loss(self.kl_weight * tf.reduce_mean(kl_divergence))

        outputs = tf.linalg.matmul(inputs, w_sample) + self.bias
        if self.activation:
            outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super(MyDenseVariational, self).get_config()
        config.update({
            'units': self.units,
            'kl_weight': self.kl_weight,
            'activation': tf.keras.activations.serialize(self.activation)
        })
        return config

    
    
    



# Shared input and layers
input_layer = Input(shape=(max_seq_length, num_features))
masked_input = Masking(mask_value=99)(input_layer)
lstm1 = LSTM(50, return_sequences=True)(masked_input)
dropout1 = Dropout(0.7)(lstm1)
lstm2 = LSTM(50, return_sequences=True)(dropout1)
dropout2 = Dropout(0.7)(lstm2)

# No-show prediction head
no_show_output = TimeDistributed(MyDenseVariational(units=1, kl_weight=0.01, activation='sigmoid'), name='no_show')(dropout2)

# Category-specific heads
medical_history_output = TimeDistributed(MyDenseVariational(units=1, kl_weight=0.01, activation='sigmoid'), name='medical_history')(dropout2)
appointment_details_output = TimeDistributed(MyDenseVariational(units=1, kl_weight=0.01, activation='sigmoid'), name='appointment_details')(dropout2)
patient_behavior_output = TimeDistributed(MyDenseVariational(units=1, kl_weight=0.01, activation='sigmoid'), name='patient_behavior')(dropout2)
temporal_variables_output = TimeDistributed(MyDenseVariational(units=1, kl_weight=0.01, activation='sigmoid'), name='temporal_variables')(dropout2)

# Combine into a single model
model = Model(inputs=input_layer, outputs=[no_show_output, medical_history_output, appointment_details_output, patient_behavior_output, temporal_variables_output])

# Custom F1 Score Metric
def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    mask = K.cast(K.not_equal(y_true, 99), 'float32')
    y_true = y_true * mask
    y_pred = y_pred * mask

    tp = K.sum(K.cast(y_true * y_pred, 'float'))
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'))
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall', 'AUC', f1_score])

# Print the model summary
model.summary()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
callbacks = [early_stopping, reduce_lr, model_checkpoint]

# Prepare the labels for the multi-output model
y_train_multi = {
    'no_show': y_train_padded['No Shows + Left without seen + cancelled_within_24h'],
    'medical_history': y_train_padded['medical_history'],
    'appointment_details': y_train_padded['appointment_details'],
    'patient_behavior': y_train_padded['patient_behavior'],
    'temporal_variables': y_train_padded['temporal_variables']
}

y_val_multi = {
    'no_show': y_val_padded['No Shows + Left without seen + cancelled_within_24h'],
    'medical_history': y_val_padded['medical_history'],
    'appointment_details': y_val_padded['appointment_details'],
    'patient_behavior': y_val_padded['patient_behavior'],
    'temporal_variables': y_val_padded['temporal_variables']
}

y_test_multi = {
    'no_show': y_test_padded['No Shows + Left without seen + cancelled_within_24h'],
    'medical_history': y_test_padded['medical_history'],
    'appointment_details': y_test_padded['appointment_details'],
    'patient_behavior': y_test_padded['patient_behavior'],
    'temporal_variables': y_test_padded['temporal_variables']
}

# Train the model
history = model.fit(X_train_padded, y_train_multi, validation_data=(X_val_padded, y_val_multi), epochs=100, batch_size=32, callbacks=callbacks)

# Evaluate the model on the test set
evaluation_results = model.evaluate(X_test_padded, y_test_multi)

# Print the test loss and all metrics
for i, metric_name in enumerate(model.metrics_names):
    print(f"{metric_name}: {evaluation_results[i]}")


def analyze_predictions_and_save_to_file(model, X_data, y_data, filename):
    predictions = model.predict(X_data)
    no_show_preds = predictions[0]
    medical_history_preds = predictions[1]
    appointment_details_preds = predictions[2]
    patient_behavior_preds = predictions[3]
    temporal_variables_preds = predictions[4]

    no_show_threshold = 0.5
    reason_threshold = 0.5  # Assuming a threshold to determine the likely reason

    with open(filename, 'w') as file:
        for i in range(len(no_show_preds)):
            for t in range(len(no_show_preds[i])):
                if no_show_preds[i][t] > no_show_threshold:  # Threshold for no-show prediction
                    reason_scores = {
                        'medical_history': medical_history_preds[i][t],
                        'appointment_details': appointment_details_preds[i][t],
                        'patient_behavior': patient_behavior_preds[i][t],
                        'temporal_variables': temporal_variables_preds[i][t]
                    }

                    # Find the reason with the highest score above the threshold
                    predicted_reason = None
                    max_score = 0
                    for reason, score in reason_scores.items():
                        if score > reason_threshold and score > max_score:
                            max_score = score
                            predicted_reason = reason

                    # If no reason is above the threshold, choose the one with the highest score
                    if predicted_reason is None:
                        predicted_reason = max(reason_scores, key=reason_scores.get)
                    
                    file.write(f"Patient {i}, Timestep {t}: No-show predicted. Likely reason: {predicted_reason}\n")

# Analyze predictions on the test set and save to a file
analyze_predictions_and_save_to_file(model, X_test_padded, y_test_multi, 'prediction_results.txt')





def analyze_predictions_and_save_to_file_aggregate(model, X_data, y_data, filename):
    predictions = model.predict(X_data)
    no_show_preds = predictions[0]
    medical_history_preds = predictions[1]
    appointment_details_preds = predictions[2]
    patient_behavior_preds = predictions[3]
    temporal_variables_preds = predictions[4]

    no_show_threshold = 0.5

    with open(filename, 'w') as file:
        for i in range(len(no_show_preds)):
            patient_reason_scores = {
                'medical_history': 0,
                'appointment_details': 0,
                'patient_behavior': 0,
                'temporal_variables': 0
            }
            no_show_count = 0

            for t in range(len(no_show_preds[i])):
                if no_show_preds[i][t] > no_show_threshold:  # Threshold for no-show prediction
                    no_show_count += 1
                    patient_reason_scores['medical_history'] += medical_history_preds[i][t]
                    patient_reason_scores['appointment_details'] += appointment_details_preds[i][t]
                    patient_reason_scores['patient_behavior'] += patient_behavior_preds[i][t]
                    patient_reason_scores['temporal_variables'] += temporal_variables_preds[i][t]

            if no_show_count > 0:
                predicted_reason = max(patient_reason_scores, key=patient_reason_scores.get)
                file.write(f"Patient {i}: No-show predicted. Likely reason: {predicted_reason}\n")

# Analyze predictions on the test set and save to a file
analyze_predictions_and_save_to_file_aggregate(model, X_test_padded, y_test_multi, 'prediction_results_aggregate.txt')





def compute_permutation_importance_for_patient(model, X_val, y_val, feature_indices, metric=f1_score, sample_size=20):
    """
    Compute permutation importance for each patient in the validation set.
    
    Parameters:
    - model: The trained model
    - X_val: Validation data
    - y_val: Validation labels
    - feature_indices: Indices of features to shuffle
    - metric: Performance metric to evaluate (default is f1_score)
    - sample_size: Number of patients to use for computing permutation importance
    
    Returns:
    - importances: Dictionary with patient index as keys and array of importance scores for each feature as values
    """
    # Ensure the correct data types
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    
    # Limit to the specified sample size
    X_val = X_val[:sample_size]
    y_val = y_val[:sample_size]
    
    # Initialize dictionary to store importances for each patient
    patient_importances = {}
    
    # Iterate over each patient
    for i in range(X_val.shape[0]):
        # Select the current patient sample
        X_sample = X_val[i:i+1]
        y_sample = y_val[i:i+1]
        
        # Compute the baseline performance
        y_pred = model.predict(X_sample)
        y_pred_flat = y_pred.flatten()
        y_sample_flat = y_sample.flatten()
        
        baseline_performance = metric(y_sample_flat, y_pred_flat)
        
        # Initialize array to store importances for the current patient
        importances = np.zeros(X_sample.shape[2])
        
        # Iterate over each feature
        for feature_idx in feature_indices:
            # Create a copy of the sample data
            X_sample_permuted = np.copy(X_sample)
            
            # Shuffle the feature values across all timesteps
            for timestep in range(X_sample.shape[1]):
                np.random.shuffle(X_sample_permuted[:, timestep, feature_idx])
            
            # Compute the performance with the shuffled feature
            y_pred_permuted = model.predict(X_sample_permuted)
            y_pred_permuted_flat = y_pred_permuted.flatten()
            
            permuted_performance = metric(y_sample_flat, y_pred_permuted_flat)
            
            # Compute the importance as the difference from the baseline
            importances[feature_idx] = baseline_performance - permuted_performance
        
        # Store the importances for the current patient
        patient_importances[i] = importances
    
    return patient_importances

# Ensure the data is in the correct format
X_val_padded = X_val_padded.astype(np.float32)
y_val_padded = y_val_padded.astype(np.float32)

# Compute permutation importance for each feature category for the sample of 5 patients
sample_size = 20
all_patient_importances = {}

for category, indices in category_feature_indices.items():
    patient_importances = compute_permutation_importance_for_patient(model, X_val_padded, y_val_padded, indices, sample_size=sample_size)
    all_patient_importances[category] = patient_importances

# Convert the importances to a DataFrame for easier printing and aggregation
patient_importances_df = {category: pd.DataFrame.from_dict(patient_importances, orient='index') 
                          for category, patient_importances in all_patient_importances.items()}

# Aggregate importances by summing across timesteps for each patient
aggregated_patient_importances_df = {category: df.sum(axis=1) for category, df in patient_importances_df.items()}


# Function to normalize importances to percentages using absolute values
def normalize_importances(aggregated_importances_df):
    normalized_importances = {}
    for patient in aggregated_importances_df['medical_history'].index:
        total_importance = sum(abs(df.loc[patient]) for df in aggregated_importances_df.values())
        if total_importance == 0:  # Handle case where total importance is zero
            normalized_importances[patient] = {category: 0 for category in aggregated_importances_df.keys()}
        else:
            normalized_importances[patient] = {category: (abs(df.loc[patient]) / total_importance) * 100 
                                               for category, df in aggregated_importances_df.items()}
    return normalized_importances

normalized_patient_importances = normalize_importances(aggregated_patient_importances_df)



# Print the model predictions and importances for each patient
for i in range(sample_size):
    print(f"Patient {i} Model Predictions: {model.predict(X_val_padded[i:i+1]).flatten()}")
    print(f"Patient {i} Category Importances:")
    for category in aggregated_patient_importances_df.keys():
        print(f"{category}:")
        print(aggregated_patient_importances_df[category].loc[i])
        print(f"Normalized {category} Importance: {normalized_patient_importances[i][category]:.2f}%")
    print("\n")

# Plot the raw and normalized importances for the sample of 20 patients
fig, axes = plt.subplots(sample_size, 2, figsize=(15, sample_size * 5))

for i in range(sample_size):
    patient_raw_data = {}
    patient_normalized_data = {}
    
    for category in aggregated_patient_importances_df.keys():
        patient_raw_data[category] = aggregated_patient_importances_df[category].loc[i]
        patient_normalized_data[category] = normalized_patient_importances[i][category]
    
    # Raw Importance Plot
    raw_df = pd.DataFrame(patient_raw_data, index=[0])
    raw_df.plot(kind='bar', ax=axes[i, 0], title=f'Patient {i} Raw Permutation Importance')
    axes[i, 0].set_ylabel('Importance')
    
    # Normalized Importance Plot (Pie Chart)
    if not np.isnan(list(patient_normalized_data.values())).all() and sum(patient_normalized_data.values()) > 0:
        axes[i, 1].pie(patient_normalized_data.values(), labels=patient_normalized_data.keys(), autopct='%1.1f%%')
        axes[i, 1].set_title(f'Patient {i} Normalized Permutation Importance')
    else:
        axes[i, 1].text(0.5, 0.5, 'No valid data', horizontalalignment='center', verticalalignment='center', transform=axes[i, 1].transAxes)
        axes[i, 1].set_title(f'Patient {i} Normalized Permutation Importance')

plt.tight_layout()
plt.show()







pred_mean, pred_stddev = predict_with_uncertainty(model, X_test_subset)
print("Predicted Means:", pred_mean)
print("Predicted Uncertainties:", pred_stddev)

# Flatten the predicted means and uncertainties
num_samples, max_seq_length, _ = pred_mean.shape
flattened_means = pred_mean.reshape(num_samples, max_seq_length)
flattened_uncertainties = pred_stddev.reshape(num_samples, max_seq_length)

# Create a DataFrame with sample indices, timesteps, means, and uncertainties
data = {
    'Sample': np.repeat(np.arange(num_samples), max_seq_length),
    'Timestep': np.tile(np.arange(max_seq_length), num_samples),
    'Predicted Mean': flattened_means.flatten(),
    'Uncertainty': flattened_uncertainties.flatten()
}
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Save to CSV if needed
df.to_csv('predicted_means_and_uncertainties.csv', index=False)





