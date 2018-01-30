# QuestionPairs - ALTEGRAD Project (MVA 2017/2018)  
Alexandre Attia, Dan Constantini, Sharone Dayan, Tom Hayat  

The aim was to predict which of the provided pairs of questions contain two questions with the
same meaning.  We tackle this challenge by considering multiple text features either from word
embedding techniques such as Word2Vec, graph information from the underlying graph, some feature
engineering techniques and finally, well-chosen classifier.

The global pipeline can be found [in this notebook](https://github.com/alexattia/QuestionPairs/blob/master/Predicting_questions_meaning.ipynb)

### Feature engineering
We introduce the different features that can arise from the analysis of the underlying data.  
We compute text mining features, embedding features, TF-IDF features and Page Rank features.  
The code to compute this pre-processing can be found in the [feature_engineering.py](https://github.com/alexattia/QuestionPairs/blob/master/feature_engineering.py) 

### Model and Comparison
We have tried different models to classify our sentences (same meaning or different meaning) :
- 1D CNN 
- Hand crafted features + Random Forest
- Hand crafted features + XGBoost
