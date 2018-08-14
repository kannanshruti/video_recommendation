# video_recommendation

## Description:
Video recommendation system using word embeddings

## Pre-requisites:
1. Datasets: clips.csv, clip_categories.csv, categories.csv
2. Python 2.7 (pip installed libraries: pandas, gensim, nltk)

## How to run:
1. Download ‘main.py’ and ‘vrec.py’ to the same folder (along with the csv files).
2. Specify the path of the folder containing the input csv files on line 16 of ‘main.py’.
    > (line 16) folder = '\<PATH>/similar-staff-picks-challenge/'
3. > python main.py

## Algorithm:
1. The solution, first, extracts all relevant information from the 3 input csv files, and combines it in a single ‘DataFrame’ for easy access. 
2. Next, words are extracted from the clip ‘title’ and clip ‘category names’ by default, and this serves as input to the word embedding model. vocabulary. Options are provided for including ‘category names’, ‘captions’ and excluding ‘stop words’ from this input ‘human’ vocabulary. 
3. Next, the model is now created using these words, with minimum occurrences of each word as 1 and the dimension of the resulting vector for each word is 10.
4. For each ‘clip ID’, the associated words in the ‘human vocabulary’ are now replaced by the average vector for the words in it.
5. For each query (‘clip ID’ for which recommendations are sought), cosine similarity is used to measure ‘similarity’ (given by high cosine value), and the top 10 of them are returned as JSON objects.

## References:
https://radimrehurek.com/gensim/
https://pandas.pydata.org/pandas-docs/stable/
pep8 was used to ensure the code files are as readable and uniform as possible
