# The Turing Testament  
A Feature Engineering Approach to Summarization.

## How to use the model 

1. Create a summariser object and instantiate the text to be summarised

    	summariser=summariser(text)

2. Make a dataframe of the text which will implicitly calculate the features and create a dataframe of them.

    	summariser.make_dataframe()
    
3. Call the function to create a summary which has two parameters.
	3.1.' max_removal'
	It is the maximum sentences which can be removed per iteration. A higher number will give a quicker summary at the cost of quality. 

    	results = summariser.summarise(max_removal= 30, x=2)

	3.2. 'x'
	This parameter defines how short the summary should be. A higher number will punish longer summaries. 


## Example Code used to extract summaries for Task-A AutoMin 

    path="Path_to_transcripts"
    summary_path="Path_to_store_summaries_in"
    filelist=os.listdir(path)
    for index,file in enumerate(filelist):
        print(index)
        for file1 in os.listdir(f'{path}\\{file}'):
            if file1[:10]=='transcript':
                with open(f'{path}\\{file}\\{file1}', encoding="utf8") as f:
                    text=f.read()

        summariser=summariser(text)
        summariser.make_dataframe()
        summary = summariser.summarise(max_removal= 30, x=2)

        # Store extractive summary
        
        with open(f'{summary_path}\\{file}.txt', 'w', encoding="utf8") as f:
            f.writelines(summary)


## Working 

The model first calculates features for each sentence in the original transcript and uses TOPSIS to rank the sentences based on these features:
1. Sentence Length 
2. Vocabulary/ Word frequency (Unigram BoW)
3. Numerical Data 
4. Topics     
5. Proper Nouns   
6. Affirmations

This is done by the function 'make_dataframe()'. 

After that the sentences are incrementally removed, and each iteration's Rouge and Meteor scores are calculated.
All these iterations are stored, and based on their length and scores, the best possible iteration is chosen as the summary. 

This ensures that the best set of sentences from the transcript are extracted as its summary.

(insert image)

    


    