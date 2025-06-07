# Détecteur de Langue : Classification de Mots en Français ou en Anglais

Ce projet est une application Python qui utilise un classificateur neuronal (`MLPClassifier`) pour identifier si un mot appartient à la langue **française** ou **anglaise**. Il s'appuie sur des dictionnaires fournis par **NLTK** et des vecteurs de caractères (`n-grams`) pour entraîner un modèle de machine learning.

## 📌 Fonctionnalités

- Apprentissage supervisé avec `MLPClassifier` (réseau de neurones à propagation avant).
- Extraction des mots depuis les dictionnaires anglais (`nltk.corpus.words`) et français (`wordnet` avec langue `fra`).
- Vectorisation des mots avec des n-grammes de caractères (bigrams et trigrams).
- Interaction en ligne de commande pour tester le modèle sur de nouveaux mots.
- Affichage de la précision du modèle sur un ensemble de test.

---

## 🚀 Installation et Exécution

### 1. Cloner le dépôt (ou copiez les fichiers dans un répertoire local)
```bash
git clone https://github.com/Fenohasina007/ExerciceGit
cd ExerciceGit
