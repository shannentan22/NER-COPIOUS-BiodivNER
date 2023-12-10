# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# %%
root_data_dir = "../Datasets/NER/BiodivNER/"

dataset = "train"
train_csv_file_path = "train.csv"
val_csv_file_path = "dev.csv"
test_csv_file_path = "test.csv"

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

# %%
data = loadData(train_csv_file_path)

# %%
val_data = loadData(val_csv_file_path)

# %%
test_data = loadData(test_csv_file_path)

# %%
VOCAB = list(set(list(data["Word"].values) + \
                 list(val_data["Word"].values) + \
                 list(test_data["Word"].values)))
VOCAB.append("ENDPAD")

n_words = len(VOCAB) #n_words includes all vocab from train and validation test.

tags = list(set(data["Tag"].values))

n_tags = len(tags)

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

# %%
tag2id = {tag: id for id, tag in enumerate(tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

# %%
tag2id_list = list(tag2id.items())

# %%
id2tag_list = list(id2tag.items())

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
import torch
import torch.nn as nn
# from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizerFast, BertForTokenClassification, T5ForConditionalGeneration, T5Tokenizer, T5TokenizerFast, AdamW, Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score

# %%
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

# %%
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
test_encodings = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

# %%
def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):      
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        #since we enabled the truncation=ture in the encoding, we should handle this again in the labels
        max_len = len(doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)]) 

        # set labels whose first offset position is 0 and the second is not 0 
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels[0:max_len]

    
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels

# %%
train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)
test_labels = encode_tags(test_tags, test_encodings)

# %%
print(len(train_encodings['offset_mapping'][0]))
print(len(train_encodings['input_ids'][0]))

# %%
for i in train_encodings["input_ids"][0:3]:
    print(len(i), i)

# %%
for i in train_labels[0:10]:
    print(len(i), i)

# %%
train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
test_encodings.pop("offset_mapping")

# %%
model = BertForTokenClassification.from_pretrained("bert-base-cased")
# model = DataParallel(model)
# print("Model Device IDs:", model.device_ids)
# optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# %%
training_args = TrainingArguments(
    output_dir="./output_dir",  # Change this to your desired output directory
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="no",
    # eval_steps=250,  # Evaluate every 250 steps
    logging_steps=100,  # Log every 100 steps
    learning_rate=5e-5,
    save_total_limit=0, 
    remove_unused_columns=True,  # Keep all columns in the dataset
    push_to_hub=False,  # Set to True if you want to push to the Hugging Face Model Hub
    gradient_accumulation_steps=4,
    # gradient_checkpointing=True,
    optim="adamw_bnb_8bit"
)

# %%
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# %%
train_dataset = NERDataset(train_encodings, train_labels)
val_dataset = NERDataset(val_encodings, val_labels)
test_dataset = NERDataset(test_encodings, test_labels)

# %%
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    # Create a long 1D list of y_true and y_pred
    y_true = []
    y_pred = []
    for preds, lbls in zip(predictions, labels):  
        [y_true.append(id2tag[l]) for p, l in zip(preds, lbls) if l != -100]
        [y_pred.append(id2tag[p]) for p, l in zip(preds, lbls) if l != -100]

    acc = accuracy_score([y_true], [y_pred])
    seqeval_report = classification_report([y_true], [y_pred])

    return {
        "accuracy": acc,
        "seqeval_report": seqeval_report
    }

# class NERDataCollator:
#     def __call__(self, batch):
#         input_ids = [item['input_ids'] for item in batch]
#         attention_mask = [item['attention_mask'] for item in batch]
#         labels = [item['labels'] for item in batch]

#         # Pad inputs and labels to the maximum sequence length in the batch
#         max_len = max(len(ids) for ids in input_ids)
#         padded_input_ids = [ids + [0] * (max_len - len(ids)) for ids in input_ids]
#         padded_attention_mask = [mask + [0] * (max_len - len(mask)) for mask in attention_mask]
#         padded_labels = [lbl + [-100] * (max_len - len(lbl)) for lbl in labels]

#         return {
#             'input_ids': torch.tensor(padded_input_ids),
#             'attention_mask': torch.tensor(padded_attention_mask),
#             'labels': torch.tensor(padded_labels),
#         }

# Create an instance of the NERDataCollator
# data_collator = NERDataCollator()

# %%
trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    # data_collator=data_collator,
    compute_metrics=compute_metrics
)

# %%
trainer.train()
eval_history = trainer.evaluate()
predictionsOutput = trainer.predict(test_dataset) 

print(eval_history["seqeval_report"])


