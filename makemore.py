from unicodedata import name
import torch
import matplotlib
names = open("names.txt" , "r").read().splitlines()


#Goes through the bigram of all chars in names, including <S>(start)
#and <E>(end) chars and every time they occur, it counts
b = {}

for w in names:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):


        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

sorted(b.items(),key = lambda kv: -kv[1])

chars = sorted(set((list(''.join(names)))))

stoi = {s:i for i,s in enumerate(chars)}
stoi['.'] = 26
print(stoi)