from summariser import summariser
import sys
import os

path=sys.argv[1]
filename=sys.argv[2]
filelist=os.listdir(path)

with open(f'{path}\\{filename}', encoding="utf8") as f:
    text=f.read()

summariser=summariser(text)
summariser.make_dataframe()
summary = summariser.summarise(max_removal= int(sys.argv[3]),  x=int(sys.argv[4]))

# Store extractive summary
with open(f'{path}\\summary_{filename}.txt', 'w', encoding="utf8") as f:
    f.writelines(summary)