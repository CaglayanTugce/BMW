#Importing required libraries
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#Reading data files
data = pd.read_csv('dataset.csv')
df=data

#Exploring Data

#First few rows of the dataframe
print(df.head())

# Basic information about the dataset: 13624 records
print(df.info())

# Number of unique part_id : 476
print("Number of unique part_id:", df['part_id'].nunique())

# Number of unique organizations: 57
print("Number of unique organizations:", df['organization'].nunique())

# Number of unique descriptions: 13163
print("Number of unique descriptions:", df['description'].nunique())

# Check for missing values :0
print(df.isnull().sum())


#Data Preprocessing

#Since descriptions are non-natural words, I'm not able to use common tools for text preprocessing. I decided to create
#tokenized descriptions by using number part of each word. Each number part act as a different word. 
def extract_numbers(description):
  return [int(num) for num in re.findall(r'ANO_(\d+)', description)]

df['tokenized_description'] = df['description'].apply(extract_numbers)

#Sequence of words could be important so I created a padded descriptions and discovered that max length of descriptions is 12 words(ANO_...)
max_length = max(map(len, df['tokenized_description']))
df['padded_description'] = df['tokenized_description'].apply(lambda x: np.pad(x, (0, max_length - len(x))))

df.head()

#Partid and organizationid are encoded. (they were object so that they could not be used in the model.)
org_encoded=LabelEncoder()
df['organization_encoded']=org_encoded.fit_transform(df['organization'])

part_encoded=LabelEncoder()
df['partid_encoded']=part_encoded.fit_transform(df['part_id'])

df.head()

#Created a dictionary to see produced  part_ids in each organization.
org_parts_df = df.groupby('organization')['part_id'].apply(lambda x: list(set(x))).reset_index()

org_parts_dict = dict(zip(org_parts_df['organization'], org_parts_df['part_id']))

#For each org_id , a new column is produced 
for org in org_parts_dict:
    df[org] = 0

#Based on the relationship between part_id and org_id matrix cells was filled with 1 if the part_id in the same row is produced in org_... 
for index, row in df.iterrows():
    part_id = row['part_id']
    for org in org_parts_dict:
        if part_id in org_parts_dict[org]:
            df.at[index, org] = 1

df.head()            
#Independent variables of the model are organization ids, description and org_... 
X_descriptions = np.vstack(df['padded_description'].values)

X = np.hstack((df[['organization_encoded']].values, X_descriptions, df.filter(like='org_').values))
y = df['partid_encoded'].values

#print("Feature matrix (X):", X)
#print("Target vector (y):", y)

#Ccreating a traning and test set, by using stratify to get balanced training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)

#To deal with class imbalance, class_weight parameter used with balanced value and also RandomOverSampling is used in the model.
model = RandomForestClassifier(class_weight='balanced', bootstrap=False)

from imblearn.over_sampling import RandomOverSampler

# Applying RandomOverSampler to the training data
ros = RandomOverSampler(sampling_strategy='auto')
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Training the model on resampled data
model.fit(X_train_resampled, y_train_resampled)

# Predictions
rosy_pred = model.predict(X_test)

rosy_pred_decoded = part_encoded.inverse_transform(rosy_pred)
y_test_decoded = part_encoded.inverse_transform(y_test)

# Evaluating the model performance : Accuracy : 0.89
print("\nRandom Forest with RandomOverSampler classification report:")
print(classification_report(y_test_decoded, rosy_pred_decoded))


