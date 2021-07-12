from .feature_extraction import feature_extraction
from math import log
import pandas as pd

class topsis(feature_extraction):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.preprocess()

    def topsis_func(self,df) ->pd.DataFrame:
        # df should only contain the feature columns. No Keys.
        # Vector normalisation
        for col in df.columns:
            mag=(sum(df[col]**2))**0.5
            if mag!=0:
                df[col]=df[col]/mag
        
        # Weights are calculated using Shannon's Entropy. 
        weights=[]
        k=1/log(len(df.columns))
        for col_index in range(len(df.columns)):
            tot=0
            for row_index in range(len(df)):
                try:
                    tot+= (df.iloc[row_index,col_index]*(log(df.iloc[row_index,col_index])))
                except:
                    continue
            tot=tot*(-k)
            weights.append(1-tot)
        
        s=sum(weights)
        weights=[i/s for i in weights]    
        for index in range(len(df.columns)):
            df.iloc[:,index]=df.iloc[:,index]*weights[index]
        
        
        df_max=df.max()
        df_min=df.min()

        
        splus=[]
        sminus=[]
        # positive impacts
        for index in range(len(df)):
            temp_min=[]
            temp_max=[]
            for i,val in enumerate(df.iloc[index]):
                temp_max.append((val-df_max[i])**2)
                temp_min.append((val-df_min[i])**2)
            
            splus.append(sum(temp_max)**0.5)
            sminus.append(sum(temp_min)**0.5)
        
        df['splus']=splus
        df['sminus']=sminus
        df['topsis']=df.sminus/(df.sminus+df.splus)
        return df 

    def make_dataframe(self) -> None:
        # convert the text into a pandas dataframe for simpler use. 
        df=pd.DataFrame()

        numerical_data_dict=self.numerical_data()
        sentence_length_dict=self.sentence_length()
        freqTable, word_frequency_dict=self.frequency()
        propernouns_dict=self.propernouns()
        affirmations_dict=self.affirmations()
        topics_score_dict=self.lda()

        df=df.append(pd.DataFrame({
            'Numerical_Data':list(numerical_data_dict.values()),
            'Sentence_length': list(sentence_length_dict.values()),
            'Word_frequency':list(word_frequency_dict.values()),
            'Proper_nouns':list(propernouns_dict.values()),
            'Affirmations':list(affirmations_dict.values()),
            'Topics':list(topics_score_dict.values())})
                                ,ignore_index=True)
        topsis_df=self.topsis_func(df)
        topsis_df['key']=list(numerical_data_dict.keys())
        self.df = topsis_df
        return topsis_df

