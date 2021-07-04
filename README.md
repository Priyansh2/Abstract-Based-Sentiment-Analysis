# About
This repository contains code for aspect-based sentiment analysis (ABSA) for the "restaurant" domain. The task is to find the sentiment polarity (positive, negative, neutral) of a given sentence corresponding to the aspect term. For instance, consider the review:- "The appetizers are ok, but the service is slow". This review/sentence has 'positive' polarity for aspect 'taste '. The polarity is 'negative' for aspect 'service.'

## Note:
The word2vec and glove are excluded from the repository and have to be download separately. `baseline.ipynb` uses word2vec and glove word-embeddings.  

## TODO:
1. Experiment with different word-embeddings approaches.
2. Experiment with architectures such as Hierarchical Attention models using Bi-GRUs, CNNs. 
3. Experiment on different standard datasets in aspect-based sentiment analysis such as "laptop customer reviews".   
