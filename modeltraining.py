import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from imblearn.over_sampling import RandomOverSampler

# Load and preprocess the data
data = pd.read_csv('dataset.csv')
df = data

def extract_numbers(description):
    return [int(num) for num in re.findall(r'ANO_(\d+)', description)]

df['tokenized_description'] = df['description'].apply(extract_numbers)

max_length = max(map(len, df['tokenized_description']))
df['padded_description'] = df['tokenized_description'].apply(lambda x: np.pad(x, (0, max_length - len(x))))

org_encoded = LabelEncoder()
df['organization_encoded'] = org_encoded.fit_transform(df['organization'])

part_encoded = LabelEncoder()
df['partid_encoded'] = part_encoded.fit_transform(df['part_id'])

org_parts_df = df.groupby('organization')['part_id'].apply(lambda x: list(set(x))).reset_index()
org_parts_dict = dict(zip(org_parts_df['organization'], org_parts_df['part_id']))

for org in org_parts_dict:
    df[org] = 0

for index, row in df.iterrows():
    part_id = row['part_id']
    for org in org_parts_dict:
        if part_id in org_parts_dict[org]:
            df.at[index, org] = 1

X_descriptions = np.vstack(df['padded_description'].values)
X = np.hstack((df[['organization_encoded']].values, X_descriptions, df.filter(like='org_').values))
y = df['partid_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

ros = RandomOverSampler(sampling_strategy='auto')
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

model = RandomForestClassifier(class_weight='balanced', bootstrap=False)
model.fit(X_train_resampled, y_train_resampled)

#Trained model and the encoders
joblib.dump(model, 'trained_model.pkl')
joblib.dump(org_encoded, 'org_encoder.pkl')
joblib.dump(part_encoded, 'part_encoder.pkl')
joblib.dump(max_length, 'max_length.pkl')
joblib.dump(org_parts_dict, 'org_parts_dict.pkl')