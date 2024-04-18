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
