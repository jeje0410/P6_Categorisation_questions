from flask import Flask, render_template, request
import string
import nltk
import pickle
import joblib
from nltk.corpus import stopwords

app = Flask(__name__, template_folder='templates', static_folder='templates/static')
model = joblib.load("model/decisionTreeClassifier.pickle")
serializer = joblib.load("model/vectorizer.pickle")
multilabel = joblib.load("model/multiLabel.pickle")


@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods= ['POST'])
def predict():
    #with open('model/decisionTreeClassifier.pickle', 'rb') as file:
    #    model =  pickle.load(file)
    title = request.form.get('title')
    print(title)
    body = request.form.get('body')
    print(body)
    print("String format required for Machine Learning prediction")
    post = title + " " + body
    post_serialize = prepocessing(post)
    prediction = model.predict(post_serialize)
    prediction = multilabel.inverse_transform(prediction)
    comptagePrediction = {}
    for x in prediction:
        for i in x:
            if (i in comptagePrediction):
                comptagePrediction[i] = comptagePrediction[i]+1
            else:
                comptagePrediction[i] = 1
    comptagePrediction = sorted(comptagePrediction.items(), key=lambda prediction: prediction[1],reverse=True)
    print(comptagePrediction)
    prediction = list(set(dict(comptagePrediction[0:1]).keys()))
    return render_template('predict.html', 
                            title = title,
                            body = body,
                            prediction_text="Tags suggested: {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)


def removeStopWord(Word_list):
    stop_words = set(stopwords.words())
    filtered_Word_list = Word_list[:] #make a copy of the Word_list
    for Word in Word_list: # iterate over Word_list
        if Word.lower() in stop_words: 
            filtered_Word_list.remove(Word) # remove Word from filtered_Word_list if it is a stopword
    return filtered_Word_list
    
#Fonction qui supprime le mot si seulement du numérique        
def removeOnlyNumeric(Word_list):
    word_list = Word_list[:] #make a copy of the Word_list
    for Word in Word_list: # iterate over Word_list
        if Word.isnumeric(): 
            word_list.remove(Word) # remove Word from filtered_Word_list if it is a stopword
    return word_list

def prepocessing(text):
    retour = ""
    retour = text.replace('\n', ' ')
    retour= retour.replace(':', '')
    punct = string.punctuation
    for c in punct:
        if c != '#':
            retour = retour.replace(c, '')
            
    retour = nltk.word_tokenize(retour,language='english')
    retour = removeStopWord(retour)
    retour = removeOnlyNumeric(retour)
    print(retour)
    retour = serializer.transform(retour)
    return retour