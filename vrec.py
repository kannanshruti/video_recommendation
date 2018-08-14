"""
Description: Class which accepts a query clip ID and gives "n" similar
    recommendations
Author: kshruti
"""

# stdlib
import numpy as np
import re
import json
from collections import Counter

# pip-installed
import pandas as pd
from gensim.models import Word2Vec
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


nltk.download('stopwords')


class VideoRecommendation:
    """
    Class implementing a video recommendation algorithm based on finding the
    average vector for the tags representing a video.

    Words in the clip title (default) and categories (optional) are used as a
    input to the algorithm, optionally, clip captions can also be included.
    """
    def __init__(self, folder, categories, clip_categories, clips):
        self.categories = pd.read_csv(folder+categories)
        self.clip_categories = pd.read_csv(folder+clip_categories)
        self.clips = pd.read_csv(folder+clips)
        self.combined_data = self.get_relevant_data()

    def get_relevant_data(self):
        """
        Returns a DataFrame with a combination of relevant columns (id,
        caption, title, category names and thumbnail).  Creates a comprehensive
        list of all clip IDs present in all DataFrames, and fills in the
        available fields from the other DataFrames.
        @param: "clips"          : DataFrame containing clip information
                "clip_categories": DataFrame mapping clips to their categories
                "categories"     : DataFrame containing category information
        @return: "combined_data": DataFrame containing columns required for generating
                          video recommendations
        """
        combined_data = pd.DataFrame(np.nan, 
                                     columns=['id', 'caption', 'title', 'category_names', 'thumbnail'], 
                                     index=range(len(self.clips.index.values)))
        df = pd.DataFrame(np.nan,
                          columns=['id', 'caption', 'title', 'category_names', 'thumbnail'],
                          index=[0])
        df['category_names'] = df['category_names'].astype(object)

        # Get relevant columns from "clips"
        combined_data['category_names'] = combined_data['category_names'].astype(object)
        combined_data['id'] = self.clips['id']
        combined_data['caption'] = self.clips['caption'].str.lower()
        combined_data['title'] = self.clips['title'].str.lower()
        combined_data['thumbnail'] = self.clips['thumbnail']
        id_list = self.clips['id'].tolist()

        # Get relevant columns from "clip_categories"
        for i in range(len(self.clip_categories['clip_id'])):
            item = self.clip_categories.iloc[i, 0]
            category_list = self.clip_categories.iloc[i, 1]
            if item in id_list:  # If clip in both clip_categories and clips
                j = id_list.index(item)
                combined_data.at[j, 'category_names'] = self.get_category_list(category_list)
            else:  # If clip is unique to clip_categories
                df['id'] = self.clip_categories.iloc[i, 0]
                df.at[0, 'category_names'] = self.get_category_list(category_list)
                combined_data = pd.concat([combined_data, df],
                                          ignore_index=True)
        return combined_data

    def get_category_list(self, category_list):
        """
        Returns category names corresponding to the category IDs
        @param: "category_list": String containing list of category IDs
        @return: "category_names": List of strings containing category names
        taken from the DataFrame "categories"
        """
        category_names = []
        if category_list == '':
            return np.nan  # If field empty
        category_list = re.sub(pattern=r'[\[\],]',
                               repl='',
                               string=category_list)
        category_id = list(category_list.split())
        for cat_id in category_id:
            j = self.categories['category_id'].tolist().index(int(cat_id))
            category_names.append(self.categories.at[j, 'name'])
        return category_names

    def get_dataset(self, column):
        """
        Returns DataFrame containing no-NaNs in the "column" of interest
        @param: "column": Column whose non-NaN rows are to be extracted
        @return: "dataset1": DataFrame containing rows of interest
        """
        dataset1 = self.combined_data.dropna(subset=[column])
        return dataset1

    def get_sentences(self, dataset1, column, include_categories=True,
                      include_caption=True, exclude_stopwords=True):
        """
        Returns list of lists containing all words in columns of interest.
        Also removes punctuations, applies stemming to the words, removes
        stop words
        @param: "dataset1"          : DataFrame whose column is to be
                    converted to a list of lists
                "column"            : (string) Column within "tests" which
                    is to be converted to a list of lists
                "include_categories": (bool) Specifies whether words from
                    categories are to be included
                "include_caption"   : (bool) Specifies whether words from
                    caption are to be included
                "exclude_stopwords" : (bool) Specifies whether stopwords are
                    to be included
        @return: "sentences": (dict, key=clipID, value=list of lists) Words
                    corresponding to clip ID
        """
        sentences = {}
        ps = PorterStemmer()
        if not exclude_stopwords:
            stop_words = set(stopwords.words('english'))
        col_names = list(dataset1.columns.values)
        categories_idx = col_names.index('category_names')
        caption_idx = col_names.index('caption')

        # Looping over the column we want to gather words from
        for i, item in enumerate(dataset1[column].tolist()):
            clip_id = int(dataset1.iloc[i, 0])
            item_list = item.decode('utf-8').split(' ')
            if include_categories and type(dataset1.iloc[i, categories_idx]) != float:
                cat_list = [c.decode('unicode-escape') for c in dataset1.iloc[i, categories_idx]]
                item_list = item_list + cat_list
            if include_caption and type(dataset1.iloc[i, caption_idx]) != float:
                item_list += dataset1.iloc[i, caption_idx].decode('utf-8').split(' ')
            if not exclude_stopwords:
                item_nostop = []
                item_nostop = [i for i in item_list if i not in stop_words]
                item_list.clear()
                item_list = item_nostop
            sentences[clip_id] = [ps.stem(re.sub(pattern=r'[\!\"#$%&\*+,-.\'\\/:;<=>?@^_`()|~=]',
                                                 repl='',
                                                 string=i)) for i in item_list]
        return sentences

    def get_model(self, sentences, min_count=1, size=10):
        """
        Returns Word Embedding model corresponding to words whose
        similarity is to be calculated
        @param: "sentences" : (dict, key=clipID, value=list of lists) Words
                    corresponding to clip ID
                "min_count" : Minimum number of times a word needs to be
                    present for it to be considered in the model
                "size"      : Size of each vector generated in the model
                    Each word in the model vocabulary will have a 'size'
                    dimension vector
        @return: model: word embedding model
        """
        sen_word_vec = list(sentences.values())
        model = Word2Vec(sen_word_vec, min_count=min_count, size=size, iter=10)
        return model

    def get_vector(self, model, sentences):
        """
        Returns the corresponding vectors for each word, along with the
        average vector for each sentence
        @param: "model"    : Word embedding model
                "sentences": (dict, key=clipID, value=list of lists) Words
                    corresponding to clip ID
        @return: "sen_vector": (dict, key=clipID, value=list of lists)
                    Vectors of each word linked to clipIDs
                 "avg_vector": (dict, key=clipID, value=list) Average vector
                    value of all words considered per clipID
        """
        sen_vector = {}
        avg_vector = {}
        for clip_id, words in sentences.items():
            tmp = []
            for word in words:
                try:
                    tmp.append(model[word])
                except:
                    continue  # If rare words
            avg_vector[clip_id] = np.mean(np.asarray(tmp), axis=0)
            sen_vector[clip_id] = tmp
        return sen_vector, avg_vector

    def get_json(self, dataset1):
        """
        Returns JSON object corresponding to each clipID in DataFrame
        @param: "dataset1": DataFrame containing the clipIDs to be converted
                    to JSON objects
        @return: "json_dict": (dict, key=clipID, value=JSOn object) Contains
                    the JSON objects corresponding to each clipID
        """
        json_dict = {}
        for i, clip_id in enumerate(dataset1['id'].tolist()):
            d = dataset1.iloc[[i]].to_dict('records')[0]
            json_dict[clip_id] = json.dumps(d, sort_keys=True)
        return json_dict

    def get_closest_match(self, query, avg_vector, n):
        """
        Returns 'n' closest clipID matches for the query clipID
        @param: "query"     : (int) clipID for similar clipIDs are to be found
                "avg_vector": (dict, key=clipID, value=list) Average vector
                    value of all words considered per clipID
                "n"         : (int) Number of matches to be found
        @return: "(query, matches)": (tuple) Contains the ''query' clipID and
                    list of 'matches' which are also clipIDs
        """
        if n > len(avg_vector)-1:
            print 'n exceeds length of dataset'
        matches_id = []
        matches = []
        try:
            query_vec = avg_vector[query]
        except KeyError:
            raise KeyError('Clip {} not found'.format(query))
        dist = Counter()
        for clip_id, vec in avg_vector.items():
            dist[clip_id] = self.cosine_sim(query_vec, vec)
        matches_id = list(reversed(dist.most_common()[:n-1]))
        for clip_id, _ in matches_id:
            matches.append(clip_id)
        return (query, matches)

    def cosine_sim(self, vec1, vec2):
        """
        Returns cosine similarity value
        @param: "vec1": (list) Vector 1
                "vec2": (list) Vector 2
        @return: cosine similarity value
        """
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)

    def get_json_list(self, json_dict, closest_match, output_file=None):
        """
        Returns JSON objects for specific clipIDs
        @param: "json_dict": (dict, key=clipID, value=JSOn object) Contains
                    the JSON objects corresponding to each clipID
                "closest_match": (list of tuples) Contains the ''query' clipID
                    and list of 'matches' which are also clipIDs
                "output_file": If the resulting JSON objects are to be written
                    as a list to a file. By default, the file is not created
        @return: "json_list": (list of dict of tuples of JSON objects)
                    Returns JSON objects corresponding to each clipID
        """
        json_list = []
        for i in closest_match:
            tmp = []
            tmp1 = {}
            for c_id in i[1]:
                tmp.append(json_dict[c_id])
            tmp1[i[0]] = tmp
            json_list.append(tmp1)
        if output_file is not None:
            with open(output_file, 'w') as file:
                json.dump(json_list, file, ensure_ascii=False)
        return json_list

    def disp_json(self, query_json):
        """
        Displays the elements in "query_json"
        @param: "query_json": List of dictionary, key is query clipID
                    values are JSON objects of similar videos
        @return: None
        """
        print 'Query:', query_json[0].keys()
        for _, v in query_json[0].items():
            for i, ele in enumerate(v):
                print i+1
                print ele, '\n'
