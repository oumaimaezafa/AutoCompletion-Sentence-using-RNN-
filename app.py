from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

app = Flask(__name__, template_folder='templates')

model =  tf.keras.models.load_model('sentence_completion.h5')
with open("tokenizer.pkl", 'rb') as file:
    tokenizer = pickle.load(file)

# Fonction d'auto-complétion de texte
def autoCompletations(text, model):
    Max_Sequence_Len=110
    # Tokenisation et vectorisation du texte
    text_sequences = np.array(tokenizer.texts_to_sequences([text]))
    # Pré-padding
    testing = pad_sequences(text_sequences, maxlen=Max_Sequence_Len-1, padding='pre')
    # Prédiction
    y_pred_test = np.argmax(model.predict(testing, verbose=0))
    
    predicted_word = ''
    for word, index in tokenizer.word_index.items():
        if index == y_pred_test:
            predicted_word = word
            break
    
    text += " " + predicted_word  # Ajoute le mot prédit sans point
    return text 

# Fonction de génération de texte pour plusieurs mots
def generate_text(text, new_words):
    for _ in range(new_words):
        text = autoCompletations(text, model)  # Ajoute chaque mot prédit au texte
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    # Si un formulaire est soumis
    if request.method == "POST":
        # Récupère les valeurs des champs de formulaire
        text = request.form.get("Text")
        no_of_words = int(request.form.get("NoOfWords"))  # Assurez-vous que c'est un entier
    
        # Générez le texte avec la fonction `generate_text`
        generated_text = generate_text(text, no_of_words)
    else:
        generated_text = ""
    
    return render_template("generate.html", output=generated_text)

# Exécution de l'application Flask
if __name__ == "__main__":
    app.run(debug=True)
