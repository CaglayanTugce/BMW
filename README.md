# BMW
## Project Overview:

This project focuses on predicting part_id based on the organization_id information and part descriptions. The primary objective is to leverage machine learning algorithms to accurately assign part_id labels to new instances based on their features.

## Algorithm Selection:

Several machine learning algorithms were explored, including Gradient Boosting and Decision Trees. After thorough experimentation, Random Forest emerged as the top-performing algorithm, delivering the most promising results in terms of predictive accuracy.

## Handling Imbalanced Classes:

To address the challenge posed by imbalanced classes within the dataset, the RandomSampleOver method was employed. This technique helps mitigate the skewed distribution of part_id labels by oversampling the minority class instances, thereby promoting a more balanced representation within the training data. Additionally, the data was stratified during the train-test split to ensure that both the training and testing sets maintain proportionate class distributions.

## Model Evaluation:

Model performance was rigorously evaluated using various metrics, with a focus on assessing both overall accuracy and class-specific performance measures such as precision, recall, and F1-score. Cross-validation techniques were applied to validate the model's generalization ability, while holdout validation ensured unbiased performance estimation on unseen data.

