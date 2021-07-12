from typing import Tuple
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from math import log10
from nltk.tag import pos_tag

import spacy
from gensim import corpora
import gensim
from spacy.lang.en import English
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer


class feature_extraction:
    def __init__(self, text: str) -> None:
        self.text= text
        ps = PorterStemmer()
        self.ps = ps
        
        spacy.load('en_core_web_sm')
        self.parser = English()
        self.en_stop = set(nltk.corpus.stopwords.words('english'))
        self.stopWords = list(set(stopwords.words("english")))
    
    def preprocess(self) -> None:
        sent_tokens = nltk.sent_tokenize(self.text)
        word_tokens = nltk.word_tokenize(self.text)
        word_tokens_lower=[word.lower() for word in word_tokens]
        word_tokens_refined=[x for x in word_tokens_lower if x not in self.stopWords]
    
        stem = []
        for w in word_tokens_refined:
            stem.append(self.ps.stem(w))
        word_tokens_refined=stem 
        self.sent_tokens= sent_tokens
        self.word_tokens= word_tokens
        self.word_tokens_lower=word_tokens_lower
        self.word_tokens_refined= word_tokens_refined

    # Features -------------------------------------------------------------------------------- 

    def frequency(self) -> Tuple[dict, dict]:
        #create vocasbulary 
        freqTable = {}
        for word in self.word_tokens_refined:
            if word in freqTable:         
                freqTable[word] += 1    
            else:         
                freqTable[word] = 1
                
        # laplace smoothing
        for k in freqTable.keys():
            freqTable[k]= log10(1+freqTable[k])
            
        #Compute word frequnecy score of each sentence
        
        word_frequency={}
        for sentence in self.sent_tokens:
            word_frequency[sentence]=0
            e=nltk.word_tokenize(sentence)
            f=[]
            for word in e:
                f.append(self.ps.stem(word))
            for word,freq in freqTable.items():
                if word in f:
                    word_frequency[sentence]+=freq
        maximum=max(word_frequency.values())
        
        for key in word_frequency.keys():
            try:
                word_frequency[key]=word_frequency[key]/maximum
                word_frequency[key]=round(word_frequency[key],3)
            except ZeroDivisionError:
                x=0
        return freqTable, word_frequency

    def numerical_data(self) -> dict:
        numeric_data={}
        for sentence in self.sent_tokens:
            numeric_data[sentence] = 0
            word_tokens = nltk.word_tokenize(sentence)
            for k in word_tokens:
                if k.isdigit():
                    numeric_data[sentence] += 1
        max_freq=max(numeric_data.values())
        for k in numeric_data.keys():
            try:
                numeric_data[k] = numeric_data[k]/max_freq
                numeric_data[k] = round(numeric_data[k], 3)
            except ZeroDivisionError:
                x=0
        return numeric_data

    def sentence_length(self) -> dict:
        sent_len_score={}
        max_score=max([len(i.split(' ')) for i in self.sent_tokens])
        for sentence in self.sent_tokens:
            sent_len_score[sentence] = (len(sentence.split(' '))/max_score)
        return sent_len_score

    def propernouns(self) -> dict:
        proper_noun={}
        for sentence in self.sent_tokens:
            tagged_sent = pos_tag(sentence.split())
            propernouns = [word for word, pos in tagged_sent if pos == 'NNP']
            proper_noun[sentence]=len(propernouns)
        maximum_frequency = max(proper_noun.values())
        
        for k in proper_noun.keys():
            try:
                proper_noun[k] = (proper_noun[k]/maximum_frequency)
                proper_noun[k] = round(proper_noun[k], 3)
            except ZeroDivisionError:
                x=0
        return proper_noun

    def affirmations(self) -> dict:
        agreed_words_list=[]
        yea_list=['ok', 'yes', 'ye', 'yeah', 'yea', 'okay','mmm', 'mmh','understood', 'k']
        for sentence in self.sent_tokens:
            words=nltk.word_tokenize(sentence)
            words=[word.lower() for word in words]
            for word in yea_list:
                if word in words and len(sentence.split(' '))<7:
                    agreed_words_list.append(1)
                    break
                elif word=='k':
                    agreed_words_list.append(0)          
        scores={}
        indices=[]
        for i in range(0,len(agreed_words_list),4):
            if sum(agreed_words_list[i:i+4])>=1:
                indices.append(list(range(i-3,i)))   # taking three sentences behind
        indices=sum(indices,[])
        for index, sent in enumerate(self.sent_tokens):
            if index in indices:
                scores[sent]=1
            else:
                scores[sent]=0
        return scores

    # Below functions are used for LDA.  
    def tokenize(self, text) -> list:
        lda_tokens = []
        tokens = self.parser(text)
        for token in tokens:
            if token.orth_.isspace():
                continue
            elif token.like_url:
                lda_tokens.append('URL')
            elif token.orth_.startswith('@'):
                lda_tokens.append('SCREEN_NAME')
            else:
                lda_tokens.append(token.lower_)
        return lda_tokens

    def get_lemma(self, word):
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma

    def get_lemma2(self, word):
        return WordNetLemmatizer().lemmatize(word)

    def prepare_text_for_lda(self, text):
        tokens = self.tokenize(text)
        tokens = [token for token in tokens if token not in self.en_stop]
        tokens = [self.get_lemma(token) for token in tokens]
        return tokens

    def lda(self) -> dict:
        # train model on the sent_tokens to get topic keywords        
        text_data = []
        for sent in self.sent_tokens:
            tokens=self.prepare_text_for_lda(sent)
            if len(tokens)>=1:
                text_data.append(tokens)
        dictionary = corpora.Dictionary(text_data)
        corpus = [dictionary.doc2bow(text) for text in text_data]
        ldamodel = gensim.models.ldamodel.LdaModel(corpus,num_topics=100, id2word=dictionary, passes=15)
        topics = ldamodel.print_topics(num_words=30)

        keywords={}
        for topic in topics:
            s=[]
            for word in range(1,len(topic[1].split('"')),2):
                if topic[1].split('"')[word] not in self.stopWords and topic[1].split('"')[word] not in keywords:
                    keywords[topic[1].split('"')[word]]=1
            
        score={}
        for sent in self.sent_tokens:
            score[sent]=0
        for topic_keys in keywords.keys():
            for sent in self.sent_tokens:
                for word in sent.split(' '):
                    if word in topic_keys:
                        score[sent]+=1
                        
        max_score=max(score.values())
        for key in score:
            score[key]=score[key]/max_score
            
        return score



