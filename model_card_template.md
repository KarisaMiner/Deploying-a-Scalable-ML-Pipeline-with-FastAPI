# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a supervised machine learning classifier made to predict whether an individual makes more than $50k per year based on demographic and employment data from the U.S. Census. The model is implemented using a RandomForestClassifier from scikit-learn. It is trained using a ML pipeline with data processing, categorical encoding, model training, and evaluation. The model, encoder, and label binarizer are stored as pickle files for later inference through FastAPI.
## Intended Use
The intended use of this model is for educational purposes. It is to assist in learning Machine Learning DevOps, by showing how to create a model with including things such as unit testing, model versioning, and API deployment.
## Training Data
The model was trained on US Census data, which contains information on individuals including demographic and employment data. The data includes numerical and categorical features. The target variable is salary with classes of over or under $50k. The dataset was split into 80% training data and 20% test data using a fixed random seed for reproducibility.
## Evaluation Data
The evaluation data consists of the 20% test split from the original dataset. The same pipeline used during training was used for the test set. To ensure consistency between training and inference, One-hot encoding of categorical features and label binarization were utilized.
## Metrics
The metrics used are Precision, Recall, and F1 Score
These measure how many predicted positive cases were actually positive, how many positive cases were correctly identified, and the mean of precision and recall. 
Precision: 0.7419 | Recall: 0.6384 | F1: 0.6863
## Ethical Considerations
This data contains demographics that pertain to race and sex. This may show differences due to historical and known racial and gender bias in employment and salary earning. This should not be used in any decision making capacity.
## Caveats and Recommendations
This data is specific to the US and is outdated.The model only shows correlations within the data, not causation. 