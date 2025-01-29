# MODE_LSTM_GR


## Description du projet
Ce projet a pour objectif de comparer les performances des modèles **GR4J** et **LSTM** dans la prévision du débit des bassins versants.
- **GR4J** est un modèle hydrologique conceptuel pluie-débit bien établi.
- **LSTM** (Long Short-Term Memory) est un modèle basé sur les réseaux de neurones récurrents (RNN), conçu pour analyser les séries temporelles.


L'étude vise à évaluer la robustesse et la précision de ces modèles en fonction des caractéristiques des bassins versants (surface, urbanisation, enneigement, etc.).


---


## Prérequis
Avant de lancer le projet, assurez-vous d'avoir installé les bibliothèques suivantes :
```bash
pip install numpy pandas matplotlib torch scikit-learn shapely geopandas
```
Si vous utilisez un environnement virtuel, créez-le et activez-le avant d'installer les dépendances :
```bash
python -m venv venv
source venv/bin/activate  # Sous Linux/Mac
venv\Scripts\activate    # Sous Windows
pip install -r requirements.txt
```
(Le fichier `requirements.txt` doit être ajouté si nécessaire.)


---


## Structure du projet
Voici les fichiers principaux et leur rôle :


### Modélisation
- **GR4J.py** : Implémente le modèle hydrologique GR4J.
- **LSTM.py** : Implémente le modèle LSTM.
- **LSTM_cas_isole.py** : Version isolée du modèle LSTM pour des tests unitaires.


### Scripts principaux
- **main_GR4J.py** : Exécute le modèle GR4J sur les données disponibles.
- **main_LSTM_training.py** : Entraîne le modèle LSTM sur les données de bassin versant.
- **main.py** : Programme central pour lancer les différentes étapes de la modélisation.


### Traitement des données et post-traitement
- **fonction.py** : Contient des fonctions utiles pour la gestion des données et la prévision.
- **fusion.py** : Fusionne les résultats des différents modèles.
- **merge_resultat.py** : Agrège les résultats des simulations.
- **read_shp.py** : Gère la lecture et l’exploitation des fichiers shapefile.
- **statistiques_BV.py** : Analyse les caractéristiques des bassins versants.
- **verif_BV_test.py** : Vérifie la cohérence des données d’entrée.


### Visualisation
- **plots.py** : Génère des graphiques pour comparer les performances des modèles.


### Données
- **data/** : Contient les fichiers CSV de données hydrologiques et météorologiques utilisées pour l’entraînement et la validation des modèles.


---


## Instructions d'exécution

### Exécuter l'ensemble des scripts
Pour exécuter le projet dans son son ensemble :
```bash
python main.py
```
Lancera si désiré la modélisation en GR4J, en LSTM et le merge des résultats et les plots.

### 1. Exécuter GR4J
Pour exécuter le modèle hydrologique GR4J :
```bash
python main_GR4J.py
```


### 2. Entraîner et tester le modèle LSTM
Pour entraîner le modèle LSTM :
```bash
python main_LSTM_training.py
```


### 3. Fusionner les résultats
Une fois les modèles exécutés, fusionner leurs résultats :
```bash
python merge_resultat.py
```


### 4. Générer des graphiques
Pour visualiser les performances :
```bash
python plots.py
```


---


## Auteurs et contributions
- Grange Emma
- Nicolas Monzie nmonzie@icloud.com
- Laveve Fabio
- Laborie Loic

Pour toute question, veuillez contacter Nicolas Monzie à l'adresse suivante : nmonzie@icloud.com
