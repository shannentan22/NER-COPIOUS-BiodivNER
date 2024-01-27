# Fine-tuning Transformer Models for NER in the Biodiversity Domain

This repository involves the fine-tuning of various transformer models for named entity recognition on two gold-standard biodiversity corpora.

## Datasets

### COPIOUS
Developed by Nhung T.H. Nguyen, Roselyn S. Gabud, and Sophia Ananiadou, COPIOUS is a corpus annotated with five entity categories related to biodiversity: taxon names, geographical locations, habitats, temporal expressions, and person names \cite{nguyen2019copious}. The corpus contains 668 documents and 28,801 entity annotations, making it suitable for training and evaluating text mining tools.
The authors evaluate the effectiveness of their corpus by using it to train CRF and BiLSTM models for NER, and by extracting binary relations between entities based on patterns.
They achieve promising results in both tasks, demonstrating the utility of the corpus for text mining in biodiversity research.

The COPIOUS corpus used in this repository was obtained from this link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6351503/

### BiodivNER
BiodivNERE, by Abdelmageed et al., is a set of gold standard corpora specifically designed for named entity recognition (NER) and relation extraction tasks. Building upon their existing ontology of core concepts (BiodivOnto), they annotated a corpus of biodiversity-focused text with entities like organisms, phenomena, and matter, along with relationships between them. The BiodivNER corpus is the part of BiodivNERE that is used for fine-tuning NER tasks.

The BiodivNER corpus used in this repository was obtained from this link: 

## Models
Seven models were involved in the fine-tuning process. Each model was trained twice—once on the COPIOUS corpus and once on the BiodivNER corpus.

### Non-BERT-based Models
1. ELECTRA: https://huggingface.co/docs/transformers/model_doc/electra
2. FNet: https://huggingface.co/docs/transformers/model_doc/fnet
3. MPNet: https://huggingface.co/docs/transformers/model_doc/mpnet

### BERT-based Models
1. BERT: https://huggingface.co/docs/transformers/model_doc/bert
2. DistilBERT: https://huggingface.co/docs/transformers/model_doc/distilbert
3. RoBERTa: https://huggingface.co/docs/transformers/model_doc/roberta

### Domain-Specific Model
1. BiodivBERT: https://ceur-ws.org/Vol-3415/paper-7.pdf
