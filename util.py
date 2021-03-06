from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score, hamming_loss,jaccard_score

#Fonction de preprocessing
stop_words = set(stopwords.words('English'))
def removeStopWord(Word_list):
    filtered_Word_list = Word_list[:] #make a copy of the Word_list
    for Word in Word_list: # iterate over Word_list
        if Word.lower() in stop_words: 
            filtered_Word_list.remove(Word) # remove Word from filtered_Word_list if it is a stopword
    return filtered_Word_list

# Instantiate stemmers
porter = PorterStemmer()
#Fonction de lemmatisation
def lemmatisation(Word_list):
    Words = Word_list[:] #make a copy of the Word_list
    Words = [porter.stem(word) for word in Words]
    return Words

#Fonction qui supprime le tag si celui ci n'appartient au TOP        
def removeNotTop100(Word_list,Top100):
    filtered_Word_list = Word_list[:] #make a copy of the Word_list
    for Word in Word_list: # iterate over Word_list
        if Word not in Top100: 
            filtered_Word_list.remove(Word) # remove Word from filtered_Word_list if it is a stopword
    if len(filtered_Word_list) == 0:
        return None
    else:
        return filtered_Word_list
    
#Fonction qui supprime le mot si seulement du numérique        
def removeOnlyNumeric(Word_list):
    word_list = Word_list[:] #make a copy of the Word_list
    for Word in Word_list: # iterate over Word_list
        if Word.isnumeric(): 
            word_list.remove(Word) # remove Word from filtered_Word_list if it is a stopword
    return word_list
    
#Fonction qui traite le C#
def processCSharp(Word_list):
    word_list = Word_list[:] #make a copy of the Word_list
    for index, value in enumerate(word_list):
        if value == '#':
            word_list.remove(value) # remove Word from filtered_Word_list if it is a stopword
            word_list[index-1] = 'c#'
    return word_list

#Fonction qui traite les termes très proches
tagSimilaire = {'c++11':'c++','angularjs':'angular','asp.net-mvc':'asp.net','datetime':'date','github':'git','java-8':'java','python-3.x':'python','visual-studio-code':'visual-studio'}
def removeTagSimilaire(Word_list):
    word_list = Word_list[:] #make a copy of the Word_list
    for index, value in enumerate(word_list):
        if value in tagSimilaire:
            if tagSimilaire[value] in word_list:
                word_list.remove(value) # remove Word
            else:
                word_list[index-1] = tagSimilaire[value]
    return word_list

# Fonction qui ne fait rien pour exploiter le pré traitement que nous avons réalisé
def dummy(doc):
    return doc

#Fonction qui renvoi différents scores afin d'évaluer les modèles
def print_score(y_test, y_pred):
    print("Hamming loss : {}".format(hamming_loss(y_test, y_pred)))
    print('Subset Accuracy : ', accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
    print('F1-score : ', f1_score(y_test, y_pred, average='micro'))
    print('Jaccard : ', jaccard_score(y_test, y_pred, average='micro'))