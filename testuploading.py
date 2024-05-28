import pandas as pd
import numpy as np
import re
import joblib
import time

# Loading the trained model and required objects
model = joblib.load('trained_model.pkl')
org_encoded = joblib.load('org_encoder.pkl')
part_encoded = joblib.load('part_encoder.pkl')
max_length = joblib.load('max_length.pkl')
org_parts_dict = joblib.load('org_parts_dict.pkl')

# Function to extract numbers from descriptions
def extract_numbers(description):
    return [int(num) for num in re.findall(r'ANO_(\d+)', description)]

# Function to preprocess the new data, assuming new data format is same with the previous given data.
def preprocess_data(df):
    df['tokenized_description'] = df['description'].apply(extract_numbers)
    df['padded_description'] = df['tokenized_description'].apply(lambda x: np.pad(x, (0, max_length - len(x))))
    df['organization_encoded'] = org_encoded.transform(df['organization'])
    
    for org in org_parts_dict:
        df[org] = 0

    for index, row in df.iterrows():
        part_id = row['part_id']
        for org in org_parts_dict:
            if part_id in org_parts_dict[org]:
                df.at[index, org] = 1
    
    X_descriptions = np.vstack(df['padded_description'].values)
    X_new = np.hstack((df[['organization_encoded']].values, X_descriptions, df.filter(like='org_').values))
    
    return X_new

# Loading new data, you can upload your file path in the pd.read_csv()
new_data = pd.read_csv('new_data.csv')

# Start timer
start_time = time.time()

# Preprocessing new data
X_new = preprocess_data(new_data)

# Making predictions
predictions = model.predict(X_new)
predictions = part_encoded.inverse_transform(predictions)

# End timer
end_time = time.time()
elapsed_time = end_time - start_time

predictions_df = pd.DataFrame(predictions, columns=['prediction'])
predictions_df.to_csv('results.csv', index=False)

print("Predictions saved to results.csv")
print(f"Total time taken: {elapsed_time:.2f} seconds")