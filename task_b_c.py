import nltk
import csv
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sentence_transformers import SentenceTransformer

class compute():
    def __init__(self, text1: str, text2: str) -> None:
        self.t1=text1
        self.t2=text2

    def process_bert_similarity(self, base_document):
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        sentences = sent_tokenize(base_document)
        base_embeddings_sentences = model.encode(sentences, show_progress_bar=False)
        base_embeddings = np.mean(np.array(base_embeddings_sentences), axis=0)
        return base_embeddings

    def compute(self):
        a=self.process_bert_similarity(self.t1)
        b=self.process_bert_similarity(self.t2)
        self.result = cos_sim([b],[a])[0][0]

    def give_result(self):
        if self.result>0.75:
            return 'TRUE'
        else:
            return 'FALSE'

import pandas as pd
import os

df=pd.DataFrame(columns=['Sl. No.', 'Instance Id', 'Predicted Label'])  

# Task B ------------------------------------------------------------------------------------------------

# path=""
# for index,file in enumerate(os.listdir(path)):
#     print('\r',index)
#     for file1 in os.listdir(f'{path}\\{file}'):
#         if file1[-11:-4]=='minutes':
#             with open(f'{path}\\{file}\\{file1}', encoding="utf8") as f:
#                 text=f.read()
#                 s=text
            
#         if file1[-14:-4]=='transcript':
#             with open(f'{path}\\{file}\\{file1}', encoding="utf8") as f:
#                 text=f.read()
#                 t=text
#     obj=compute(t,s)
#     obj.compute()
#     res=obj.give_result()
#     df=df.append({
#     'Sl. No.':index+1,
#     'Instance Id': file,
#     'Predicted Label': res}, ignore_index=True)

# res_path=''

# df.to_csv(res_path, sep='\t', index=False)

# Task-C--------------------------------------------------------------------------------------------------

# path=""
# for index,file in enumerate(os.listdir(path)):
#     print('\r',index)
#     for file1 in os.listdir(f'{path}\\{file}'):
#         if file1[-5]=='A':
#             with open(f'{path}\\{file}\\{file1}', encoding="utf8") as f:
#                 text=f.read()
#                 s=text
            
#         if file1[-5]=='B':
#             with open(f'{path}\\{file}\\{file1}', encoding="utf8") as f:
#                 text=f.read()
#                 t=text
#     obj=compute(t,s)
#     obj.compute()
#     res=obj.give_result()
#     df=df.append({
#     'Sl. No.':index+1,
#     'Instance Id': file,
#     'Predicted Label': res}, ignore_index=True)

# res_path=''

# df.to_csv(res_path, sep='\t', index=False)