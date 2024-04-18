# CS4248-NLP_Project

Find the raw data here:
https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection/releases/tag/dataset

**NOTE**: Don't push the raw data to the repo since it is too big.

All data files find them here: https://drive.google.com/drive/u/2/folders/1QDKXsqweW7tyCg5POcMpOFvjp-PybpOn

## For Reviewers

Do come to this google drive link to find all the data files we used. There is a memory space limit for github repository.
https://drive.google.com/drive/u/2/folders/1QDKXsqweW7tyCg5POcMpOFvjp-PybpOn

Here is the folder structure
- **DATASET FOR ALL TRAINING MODELS**: This is the file used to train and test the model. It is a file you will be interested in to run the models.
- **NER_POS_data_file (rejected)**: This is a file for NER and POS processing that we experimented that did not made it to the final draft
- **preprocessed_data_jl**: This is the folder with preprocessed data we had before feature engineering
- **raw_data_only_particular_classification: folder that contain data files for fulltrain.csv but separated into the 4 classifications (satire, )
- **raw_data_original**: the original data files
- **results**: feature importance interpretation of the trained models for each classification. 
  - logistic regression: coefficients
  - naive bayes: log probability
  - RNN & LSTM: Gradients

Step 0: Install the necessary packages by running the cell under *Installing necessary libraries*.
```
!pip install scikit-learn
!pip install pandas
!pip install matplotlib
!pip install tensorflow
!pip install shap
!pip install nltk
!pip install textstat
!pip install textblob
```
Step 1: Run The following cell under *Constant*.
```
import pandas as pd
# Data consists of text (feature) + classification (y value: 1/2/3/4)
TEXT_FEATURE_NAME = "text"
CLASSIFICATION_NAME = "classification"

# CONSTANTS
classifier_mapping = {
    1: "Satire",
    2: "Hoax",
    3: "Propaganda",
    4: "Reliable News"
}
mapping_df = pd.DataFrame(list(classifier_mapping.items()), columns=['classification', 'label'])
```
Step 2: Find this subheader *Reading the Data (No need care about above if we already have normalised data)* under header *Model Training & Evaluation*. Edit the `preprocessed_data/normalised_fulltrain.csv` to match the file path of the training file. As mentioned, the data file is in the google drive link under the folder **DATASET FOR ALL TRAINING MODELS**

```
import pandas as pd

normalised_df = pd.read_csv('preprocessed_data/normalised_fulltrain.csv')
X = normalised_df.drop(columns=[CLASSIFICATION_NAME])
y = normalised_df[CLASSIFICATION_NAME]
```

Step 3: Run the code under `Setting for train test split` subheader.

## Settings
```
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

## Note: Change this to fit the algorithm below
# X are the features
X = X
# y are the outputs
y = y
# test_size is the size of the test (0 < test_size < 1)
test_size = 0.2
# seed for random split
seed = 40
## End of Note

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
```

Step 4: Run a particular model. In this case, I will be running this cell under `Logistic Regression`. You will see the output of the confusion matrix and the F1 score and accuracy
```
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

evaluation_metrics = [("Accuracy", accuracy_score), ("Confusion Matrix", confusion_matrix)]

for evaluation_metric_name, evaluation_metric_func in evaluation_metrics:
    print(f"{evaluation_metric_name}:\n{evaluation_metric_func(y_test, y_pred)}")
print(f"F1 Macro Score: {f1_score(y_test, y_pred, average='macro')}")
```

## For Developers

Sample .gitignore file
`nlpenv` is my virtual environment.
```
nlpenv/
.gitignore
.ipynb_checkpoints/
raw_data/
balancedtest.csv
```

Sample Folder Directory (Mine)

```
>.ipynb_checkpoints
> nlpenv
> raw_data
    - balancedtest.csv
    - fulltrain.csv
    - test.xlsx
    - xdev.txt
    - xtrain.txt
- .gitignore
- balancedtest.csv
- README.md
- Team18.ipynb
```
