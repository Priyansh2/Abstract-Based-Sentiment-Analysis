# NLP FINAL PRJECT (ABSA)
This repository contains code for aspect based sentiment analysis (ABSA) for "restaurant" domain. The task is to find sentiment polarity (postive, negative, neutral) of given sentence with its aspect-term.
For instance, consider the review :- "The appetizers are ok, but the service is slow". This review/sentence has 'positive' polarity for aspect 'taste' and the polarity is 'negative' for aspect 'service'.

## Note:
The word2vec and glove are not included in the repo. and has to be download
separatly. `baseline.ipynb` uses word2vec and glove word-embeddings.  

## TODO:
1. Experiment with custom word-embeddings approaches.
2. Experiment with more advanced architectures like Hierarchical Attention models using Bi-GRUs,CNNs. 
3. Experiment on other common datasets in aspect-based sentiment analysis like "laptop cutomer reviews".  
