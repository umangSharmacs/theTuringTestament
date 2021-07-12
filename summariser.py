import os

from rouge_score import rouge_scorer
from math import log
from numpy import std as standard_deviation
from nltk.translate.meteor_score import single_meteor_score

from .topsis import topsis

class summariser(topsis):
    def summarise(self, max_removal, x) -> list:
        index=0
        slopes=[]
        r=[]              # contains the min and max values for each iteration
        dist=[0,0]           # range for all the iterations
        df_list=[]        # contains all the dataframes
        transcript=[]
        length=[]
        # get original transcript 
        for sent in self.df.key:
            transcript.append(sent)
        original_transcript=''.join(transcript)
        df_list.append(self.df.drop([ 'key'], axis=1))
        length.append(len(df_list[-1]))
        
        _df=df_list[-1]
        r_scores_summ=[]
        r_scores_transcript=[]
        m_scores_transcript=[]
        m_scores_summ=[]
        check=False            # bool value that will check slope of dist and appropriately change the number of sentences removed.
        sentence_remove_threshold=(max_removal//2)  # number of sentences to remove
        
        index_check=0
        while len(_df)>10:  # Minimum 10 sentences will be in the summary
            #loader = Loader(f"Summarising text... {((length[0]-length[-1])*100)/length[0]} % done", "   That was fast !!!", 0.5).start()
            print(f"Summarising text... {((length[0]-length[-1])*100)/length[0]} % done", flush=True, end='\r')
            if sentence_remove_threshold<1:
                sentence_remove_threshold=1
            if index_check<5:
                index_check+=1
                check=False
            elif index_check==5:
                index_check=0
                check=True

            # TODO    
            #print('Prev removed: ', sentence_remove_threshold)
            
            _df=df_list[-1]
            _df=_df.sort_values('topsis', ascending=False)
            
            if check is True:
                std=standard_deviation(slopes[-5:])
                if std<=0.01 and sentence_remove_threshold<max_removal:
                    sentence_remove_threshold+=(max_removal//2)
                elif std>0.1 and sentence_remove_threshold>(max_removal//2):
                    sentence_remove_threshold-=(max_removal//2)
            
            _df=_df.drop(_df.iloc[-sentence_remove_threshold:].index)
            df_new=self.topsis_func(_df)
            df_list.append(df_new)
            r.append((df_new.topsis.min(), df_new.topsis.max()))
            dist.append(df_new.topsis.max()-df_new.topsis.min())
            slope=dist[-1]-dist[-2]/sentence_remove_threshold
            length.append(len(df_new))
            slopes.append(slope)  
            df_new=df_new.sort_index()
            df_new['key']=self.df['key']
            summ=[]
            for sent in df_new.key:
                summ.append(sent)
            topsis_summary=''.join(summ)
            
            #rscorer = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeL'], use_stemmer=True)
            rscorer = rouge_scorer.RougeScorer(['rouge1','rouge2'], use_stemmer=True)
            
            #Rouge vs original transcript 
            t=rscorer.score(topsis_summary,original_transcript)
            r_scores_transcript.append(t)
            
            #meteor vs transcript
            t=single_meteor_score(original_transcript,topsis_summary)
            m_scores_transcript.append(t)
            
         
        r1=[i['rouge1'][-1] for i in r_scores_transcript[:-1]]
        r2=[i['rouge2'][-1] for i in r_scores_transcript[:-1]]
        #rl=[i['rougeL'][-1] for i in r_scores_transcript[:-1]]
        
        vals=[log(r1[i])+log(r2[i])+log(m_scores_transcript[i])-x*log(length[i]) for i in range(len(r1))]
        max_val=vals.index(max(vals))
        df_list[max_val]['key']=self.df['key']
        summ=[sent for sent in df_list[max_val].sort_index()]
        summ=''.join(summ)
        self.extractive_summary=summ
        return summ





# path="Path_to_transcripts"
# summary_path="Path_to_store_summaries_in"
# filelist=os.listdir(path)
# for index,file in enumerate(filelist):
#     print(index)
#     for file1 in os.listdir(f'{path}\\{file}'):
#         if file1[:10]=='transcript':
#             with open(f'{path}\\{file}\\{file1}', encoding="utf8") as f:
#                 text=f.read()

#     summariser=summariser(text)
#     summariser.make_dataframe()
#     summary = summariser.summarise(max_removal= 30, x=2)

#     # Store extractive summary
    
#     with open(f'{summary_path}\\{file}.txt', 'w', encoding="utf8") as f:
#         f.writelines(summary)
    
