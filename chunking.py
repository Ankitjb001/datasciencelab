import nltk
# nltk.download()
text="program to implement k-nn classification using any standard dataset"
wordtok=nltk.word_tokenize(text)
print(wordtok)
wordtag=nltk.pos_tag(wordtok)
print(wordtag)
grammar="NP:{<DT>?<JJ>*<NN>}"
st=nltk.RegexpParser(grammar)
wt=st.parse(wordtag)
print(wt)
wt.draw()