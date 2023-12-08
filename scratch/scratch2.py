# %% [markdown]
# # BiodivNER Data Preprocessing

# %% [markdown]
# Adapted from: Nora Abdelmageed's BiodivBERT (2023) <br>
# Original code: https://github.com/fusion-jena/BiodivBERT/tree/main

# %% [markdown]
# ## Import and Configurations

# %%
import subprocess
packages = ['pandas', 'numpy', 'matplotlib', 'tabulate']
for p in packages:
   subprocess.run(['pip', 'install', p])

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# %% [markdown]
# ## Dataset Configurations

# %%
root_data_dir = "./Datasets/NER/BiodivNER/"

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
val_data = loadData(val_csv_file_path)

# %%
test_data = loadData(test_csv_file_path)

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
# ## Encoder format

# %% [markdown]
# Encoder/decoder dictionaries for tags

# %%
tag2id = {tag: id for id, tag in enumerate(tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

# %%
tag2id_list = list(tag2id.items())

# %%
id2tag_list = list(id2tag.items())

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

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Convert IOB2 tags to T5 format
def convert_iob2_to_t5(words, tags):
    t5_labels = []

    for word, tag in zip(words, tags):
        if tag.startswith('B-'):
            t5_labels.append(f"<{tag[2:]}> {word}")
        elif tag.startswith('I-'):
            t5_labels.append(word)
        else:
            t5_labels.append(f"</{tag[2:]}>")

    return ' '.join(t5_labels)

# Apply the conversion to all sentences in your dataset
train_t5_inputs = [convert_iob2_to_t5(words, tags) for words, tags in zip(train_texts, train_tags)]
val_t5_inputs = [convert_iob2_to_t5(words, tags) for words, tags in zip(val_texts, val_tags)]
test_t5_inputs = [convert_iob2_to_t5(words, tags) for words, tags in zip(test_texts, test_tags)]

# Tokenize training, validation, and test data
def tokenize_data(input_texts):
    input_ids = []
    attention_masks = []

    for text in input_texts:
        encoded = tokenizer.encode_plus(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

# Tokenize training, validation, and test data
X_train, attention_masks_train = tokenize_data(train_t5_inputs)
X_val, attention_masks_val = tokenize_data(val_t5_inputs)
X_test, attention_masks_test = tokenize_data(test_t5_inputs)

# Assuming you have labels in the same IOB2 format
y_train, _ = get_text_tags_lists(sentences)
y_val, _ = get_text_tags_lists(sentences_val)
y_test, _ = get_text_tags_lists(sentences_test)

# Convert labels to IDs using the tokenizer
def convert_labels_to_ids(labels):
    label_ids = [tokenizer.encode(label, return_tensors='pt')[0] for label in labels]
    return torch.cat(label_ids, dim=0)

# Convert labels to IDs
y_train_ids = convert_labels_to_ids(y_train)
y_val_ids = convert_labels_to_ids(y_val)
y_test_ids = convert_labels_to_ids(y_test)

# Create DataLoader
train_dataset = TensorDataset(X_train, attention_masks_train, y_train_ids)
val_dataset = TensorDataset(X_val, attention_masks_val, y_val_ids)
test_dataset = TensorDataset(X_test, attention_masks_test, y_test_ids)

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model and optimizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./output_dir",  # Change this to your desired output directory
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=500,  # Save model every 500 steps
    evaluation_strategy="steps",
    eval_steps=250,  # Evaluate every 250 steps
    logging_steps=100,  # Log every 100 steps
    learning_rate=5e-5,
    save_total_limit=2,  # Only keep the last two models
    remove_unused_columns=False,  # Keep all columns in the dataset
    push_to_hub=False,  # Set to True if you want to push to the Hugging Face Model Hub
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=val_dataloader,
)

# Train the model
trainer.train()

# Evaluate on the test set
results = trainer.evaluate(test_dataloader)
print(results)

# Testing
model.eval()
test_predictions = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Testing"):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=128)
        # Decode the generated tokens
        preds = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        test_predictions.extend(preds)

# Convert predictions and labels to IOB2 format for seqeval
flat_test_predictions = [word for pred in test_predictions for word in pred.split()]
flat_test_labels = [word for label in y_test for word in tokenizer.decode(label, skip_special_tokens=True).split()]

# Print classification report for the test set using seqeval
print(classification_report([flat_test_labels], [flat_test_predictions]))