from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import words, wordnet
import random

# Télécharger les ressources nécessaires de NLTK si pas encore fait
nltk.download('words')
nltk.download('omw-1.4')
nltk.download('wordnet')

def charger_dictionnaires():
    """Charge les dictionnaires de mots pour le français et l'anglais."""
    mots_anglais = set(mot.lower() for mot in words.words())
    mots_francais = set(mot.lower() for synset in wordnet.all_synsets()
                        for mot in synset.lemma_names('fra'))
    return mots_anglais, mots_francais

def valider_modele(donnees_entrainement, etiquettes):
    """Valide le modèle sur un ensemble de test."""
    donnees_entrain, donnees_test, etiquettes_entrain, etiquettes_test = train_test_split(
        donnees_entrainement, etiquettes, test_size=0.2, random_state=42
    )
    
    # Modèle avec 6 perceptrons dans une couche cachée
    classificateur = MLPClassifier(hidden_layer_sizes=(6,), max_iter=1000, random_state=42)
    classificateur.fit(donnees_entrain, etiquettes_entrain)
    
    predictions_test = classificateur.predict(donnees_test)
    precision = accuracy_score(etiquettes_test, predictions_test)
    print(f"Précision sur l'ensemble de test : {precision * 100:.2f}%")
    return classificateur

def principal():
    # Charger les dictionnaires
    mots_anglais, mots_francais = charger_dictionnaires()
    
    # Créer un ensemble de données pour l'entraînement
    mots_entrainement = random.sample(list(mots_anglais), 1000) + random.sample(list(mots_francais), 1000)
    etiquettes = [1] * 1000 + [0] * 1000  # 1 pour anglais, 0 pour français
    
    # Vectoriser les mots avec des n-grammes de caractères
    vectoriseur = CountVectorizer(analyzer='char', ngram_range=(2, 3))
    donnees_entrainement = vectoriseur.fit_transform(mots_entrainement)
    
    # Valider et entraîner le classificateur
    print("Entraînement et validation du classificateur...")
    classificateur = valider_modele(donnees_entrainement, etiquettes)
    print("Entraînement terminé.")
    
    print("\n\n-----------------------------------------------------------------\n")
    print("\nDÉTECTEUR ET CLASSIFICATION DE MOT [FRANÇAIS] OU [ANGLAIS] !\n")
    print("\n-----------------------------------------------------------------\n\n")
    print("Entrez un mot (ou 'quitter' pour arrêter).")
    
    while True:
        entree_utilisateur = input("Veuillez entrer votre mot : ")
        if entree_utilisateur.lower() == 'quitter':
            print("Programme terminé.")
            break
        
        if not entree_utilisateur.strip():
            print("Veuillez entrer un mot valide.")
            continue
        
        caracteristiques = vectoriseur.transform([entree_utilisateur.lower()])
        prediction = classificateur.predict(caracteristiques)[0]
        resultat = "Anglais" if prediction == 1 else "Français"
        print(f"Résultat : {resultat} (prédit par le classificateur)\n")

if __name__ == "__main__":
    principal()
