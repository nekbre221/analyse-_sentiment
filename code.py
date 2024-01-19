import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.language import Language
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

nlp = spacy.load('fr_core_news_sm')
"""
1. Télécharger les données présent à l’adresse suivante qui représentant les commentaires :
https://drive.google.com/file/d/1BsN1lX4i1AnUDe8iOsSb93VNelQyln4c/view?usp=share
_link   
    Ici nous allons importater de la BD dans un objet appelé base a l'aide de la library pandas
"""

base=pd.read_csv("data_set.csv")





"""
2. Réaliser le prétraitement de ces données

ici il est question pour nous principalement de supprimer la variable Id_Commentaire
qui n'est d'aucune importance pour notre analyse
"""

del base["Id_Commentaire"]

#base.dtypes

"""
nous constatons aussi qu'il se trouve dans notre BD des labels + et - qui ne peuvent être 
interpreter par le compilateur. Nous passons donc au recodage des modalités avec la legende
ci dessous

                - = 0 (insatisfait)   et    + = 1 (satisfait)
"""



# Définition du mapping des couleurs
mapping = {'-': 0, '+': 1}

# Recodage de la colonne "Couleur"
base['Sentiments'] = base['Sentiments'].map(mapping)
base.head()



# Téléchargement des ressources nécessaires (exécutez cette ligne une seule fois)

nltk.download('punkt')
nltk.download('stopwords')

#Définissons les fonctions de prétraitement :
    
    
def preprocess_text(text):
    
    # Tokeniser le texte en mots
    tokens = word_tokenize(text)
    
    #Suppression des ponctuations
    tokens2=list( filter(lambda token: nltk.tokenize.punkt.PunktToken(token).is_non_punct, tokens))
    
    # Conversion en minuscules
    tokens3 = [nekui.lower() for nekui in tokens2]
    
    # Suppression des mots vides
    tokens4=list( filter(lambda pit: pit not in stopwords.words("french"),tokens3)) 
    
    #racinisation ou stemming
    stemmer=SnowballStemmer("french")
    tokens5=[stemmer.stem(nek) for nek in tokens4]

    #lemmatisation
    # Analyse du texte avec spaCy
    
    tokens6 = nlp(" ".join(tokens4))

    return tokens6


# Appliquons la fonction de prétraitement sur votre dataset 
base['preprocessed_text'] = base['Commentaires'].apply(preprocess_text)

# ma nouvelle base
new_base=pd.DataFrame({"Commentaires":base.preprocessed_text,"Sentiments":base.Sentiments})

X = base['Commentaires']
y = base['Sentiments']


"""
3. Subdiviser l’ensembles des données en données d’entrainement (80%) et en données de
test (20%)
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


"""
4. Construire un modèle de classification en utilisant l'algorithme de votre choix (Naïve Bayes
ou Knn)."""


# Créer une instance de CountVectorizer
vectorizer = CountVectorizer()

# Convertir les textes en une matrice de compte de mots
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)



classifier = MultinomialNB()
# Entraîner le modèle de classification
classifier.fit(X_train_count, y_train)


# Prédire les étiquettes pour les données de test
y_pred = classifier.predict(X_test_count)




"""
 5. Générer la matrice de confusion du modèle de classification construit
"""

confusion=confusion_matrix(y_test, y_pred)




"""
# 6. Imprimer les rapports de la classification puis évaluer la performance du modèle utilisé.
"""

rapport= classification_report(y_test,y_pred,zero_division=True)

# Calculer la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
accuracy





"""
7. Utiliser votre modèle entraîné pour prédire le sentiment des commentaires non étiquetés
suivants :
- "Ce produit est incroyable !"
- "Je suis déçu de mon achat."
- "Le service client est exceptionnel."
- " Vous êtes le meilleur
"""

a=["Ce produit est incroyable !", "Je suis déçu de mon achat.","Le service client est exceptionnel."," Vous êtes le meilleur"]

aa=pd.DataFrame({"Commentaires":a})

#prétraitons le vecteur avec notre fonction preprocess_text du haut et appelons leux aa

aa['Commentaires'] = aa['Commentaires'].apply(preprocess_text)

vectorizer = CountVectorizer()
aa_count = vectorizer.fit_transform(X_train)
predict = classifier.predict(aa_count)
predict
prediction=pd.DataFrame({"Sentiments":aa,"prediction":predict[0,1]})


print("Sentiment prédit: pour Ce produit est incroyable !   :", predict[0])
print("Sentiment prédit: pour Je suis déçu de mon achat.     :", predict[1])
print("Sentiment prédit: pour Le service client est exceptionnel.   :", predict[2])
print("Sentiment prédit: pour Vous êtes le meilleur        :",predict[3])
# soit 50% de bonne prediction!!
