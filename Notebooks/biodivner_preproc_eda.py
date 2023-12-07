# %% [markdown]
# # BiodivNER Data Preprocessing

# %% [markdown]
# Adapted from: Nora Abdelmageed's BiodivBERT (2023) <br>
# Original code: https://github.com/fusion-jena/BiodivBERT/tree/main

# %% [markdown]
# ## Import and Configurations

# %%
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install tabulate

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# %% [markdown]
# ## Dataset Configurations

# %%
root_data_dir = "../Datasets/NER/BiodivNER/"

dataset = "train"
train_csv_file_path = "train.csv"
val_csv_file_path = "dev.csv"
test_csv_file_path = "test.csv"

# %% [markdown]
# ## Data Loading Utilities

# %%
def loadData(csv_file_path):
  dataset_path = os.path.join(root_data_dir, csv_file_path)
  data = pd.read_csv(dataset_path, encoding="latin1")
  data = data.fillna(method="ffill")
  return data

# %%
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),                                                          
                                                        s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

# %% [markdown]
# ## Load Datasets

# %%
data = loadData(train_csv_file_path)

# %%
data.head(10)

# %%
val_data = loadData(val_csv_file_path)

# %%
val_data.head(10)

# %%
test_data = loadData(test_csv_file_path)

# %%
test_data.head(10)

# %% [markdown]
# ## Exploratory Data Analysis

# %% [markdown]
# ### Vocabulary and tags

# %%
#if the data are in string style, we propably use tokenzer.fit_on_texts instead of list manipulation like here

VOCAB = list(set(list(data["Word"].values) + \
                 list(val_data["Word"].values) + \
                 list(test_data["Word"].values)))
VOCAB.append("ENDPAD")

n_words = len(VOCAB) #n_words includes all vocab from train and validation test.

tags = list(set(data["Tag"].values))

n_tags = len(tags)

# %%
print(n_words, VOCAB[0:10] ,VOCAB[n_words-1], sep='\n')

# %%
print(n_words, VOCAB[0:10] ,VOCAB[n_words-1], sep='\n')

# %% [markdown]
# ### Creating sentences for train, val, and test sets

# %%
getter = SentenceGetter(data)
sentences = getter.sentences
sent = getter.get_next()
print(sent)

# %%
getter_val = SentenceGetter(val_data)
sentences_val = getter_val.sentences
sent_val = getter_val.get_next()
print(sent_val)

# %%
getter_test = SentenceGetter(test_data)
sentences_test = getter_test.sentences
sent_test = getter_test.get_next()
print(sent_test)

# %% [markdown]
# ### Sentence lengths

# %% [markdown]
# T5's recommended number of tokens per input is at most 512. The purpose of this is to save on memory resources.

# %%
plt.style.use("ggplot")

# %%
plt.hist([len(s) for s in sentences_val], bins=5)
plt.show()

# %%
max([len(s) for s in sentences_val])

# %% [markdown]
# ## Encoder format

# %% [markdown]
# Encoder/decoder dictionaries for tags

# %%
tag2id = {tag: id for id, tag in enumerate(tags)}
id2tag = {id: tag for tag, id in tag2id.items()}
print(tag2id)

# %%
tag2id_list = list(tag2id.items())
print(tabulate(tag2id_list, headers=['Tag', 'ID']))

# %%
id2tag_list = list(id2tag.items())
print(tabulate(id2tag_list, headers=['ID', 'Tag']))

# %% [markdown]
# Split texts from tags (use two different Python lists)

# %%
def get_text_tags_lists(sentences):
  texts = []
  tags = []
  for sent in sentences: #list of tuples    
    sent_texts = []
    sent_tags = []  
    for tuple1 in sent:  
      sent_texts.append(tuple1[0])
      sent_tags.append(tuple1[1])

    texts.append(sent_texts)
    tags.append(sent_tags)
  return texts, tags

# %%
train_texts, train_tags = get_text_tags_lists(sentences)
val_texts, val_tags = get_text_tags_lists(sentences_val)
test_texts, test_tags = get_text_tags_lists(sentences_test)

# %%
for row in train_texts[0:9]:
    print(row)

# %%
for row in train_tags[0:9]:
    print(row)


