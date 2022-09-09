from unicodedata import name


names = open("names.txt" , "r").read().splitlines()

print(names[:10])


#Goes through the bi gram of all chars in names, including <S>(start)
#and <E>(end) chars
b = {}
for w in names[:1]:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
        print(ch1, ch2)
