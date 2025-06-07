# 🧠 Détecteur de Langue : Classification de Mots en Français ou en Anglais

Ce projet est une application Python qui utilise un réseau de neurones (`MLPClassifier` de `scikit-learn`) pour prédire si un mot appartient à la langue **française** ou **anglaise**. Il s’appuie sur les dictionnaires fournis par **NLTK** et transforme les mots en vecteurs d’**n-grammes de caractères**.

---

## 🚀 Fonctionnalités

- Prédiction de la langue d’un mot unique : **Français** ou **Anglais**.
- Entraînement automatique sur 2000 mots issus des dictionnaires NLTK.
- **Précision moyenne du modèle : ~85%** sur l’ensemble de test.
- Interface en ligne de commande pour tester vos propres mots.

---

## 📈 Précision du modèle

Le modèle utilise un **réseau de neurones à 1 couche cachée de 6 neurones**, entraîné sur un jeu de données équilibré (1000 mots français + 1000 mots anglais).

### ⚙️ Performance :
- **Taux de précision moyen : 85%**
- Évalué automatiquement sur un sous-ensemble de test (20%).

> Ce taux peut légèrement varier à chaque exécution selon l’échantillonnage aléatoire.

---


## 🚀 Installation et Exécution

### 1. Cloner le dépôt (ou copiez les fichiers dans un répertoire local)
```bash
git clone https://github.com/Fenohasina007/ExerciceGit
cd ExerciceGit
