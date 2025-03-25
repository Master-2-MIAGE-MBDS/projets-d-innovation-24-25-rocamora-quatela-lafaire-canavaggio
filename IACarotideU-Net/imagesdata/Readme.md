Projet de Détection des Carotides par U-Net
Vue d'ensemble
Ce projet utilise l'architecture U-Net pour détecter les carotides dans des images échographiques. Il s'agit d'un outil de segmentation d'images médicales qui peut aider les professionnels de la santé à identifier et délimiter précisément les carotides dans les échographies.
Fonctionnalités

Prétraitement des images échographiques
Génération de masques binaires pour les carotides
Entraînement de modèles U-Net pour la segmentation d'images
Augmentation de données pour améliorer la robustesse du modèle
Validation croisée pour une meilleure évaluation de performance
Mode test pour prédire sur de nouvelles images
Visualisation des résultats avec superposition des masques prédits

Prérequis

Python 3.7 ou supérieur
TensorFlow 2.x
OpenCV
NumPy
Matplotlib
scikit-learn
Albumentations (pour l'augmentation de données)

Installation
bashCopier# Cloner le dépôt
git clone [URL du dépôt]

# Installer les dépendances
pip install -r requirements.txt
Structure du projet
La structure des dossiers du projet est la suivante :
CopierIACarotideU-Net/
├── imagesdata/
│   ├── images/                # Images échographiques d'entrée
│   ├── masks/                 # Masques d'annotation originaux
│   ├── corrected_masks/       # Masques corrigés pour l'entraînement
│   ├── predictions/           # Résultats de prédiction
│   ├── models/                # Modèles entraînés
│   ├── results/               # Résultats d'évaluation
│   └── test_images/           # Images pour le test
Configuration
Avant d'exécuter le script, assurez-vous de configurer correctement les chemins de dossiers dans le fichier principal:
pythonCopier# Configuration
BASE_DIR = "/chemin/vers/votre/dossier/IACarotideU-Net/imagesdata"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "corrected_masks")
Utilisation
Mode Entraînement
Pour entraîner un nouveau modèle:
bashCopierpython main.py --mode train
Mode Test
Pour tester un modèle existant sur de nouvelles images:
bashCopierpython main.py --mode test --model /chemin/vers/votre/modele.h5 --test_dir /chemin/vers/dossier/images_test
Modules du projet
Le projet comprend plusieurs scripts avec des fonctionnalités distinctes:

U-Net principal - Entraînement et test du modèle de segmentation
Détection des annotations - Crée des masques binaires à partir d'images annotées
Augmentation de données - Génère des variations d'images pour améliorer l'entraînement
Version améliorée - Version optimisée avec validation croisée et métriques avancées

Détails techniques
Architecture U-Net
Le modèle utilise une architecture U-Net avec:

Encodeur à 4 niveaux avec convolutions, BatchNormalization et Dropout
Décodeur avec correspondance des caractéristiques via concatenate
Fonction de perte combinée (BCE pondérée et coefficient Dice)

Métriques d'évaluation

Coefficient Dice
Sensibilité (Recall)
Spécificité
IoU (Intersection over Union)

Augmentation des données
Les techniques d'augmentation incluent:

Rotations (±20°)
Miroir horizontal
Variations de contraste/luminosité
Zoom léger (90-110%)
Ajout de bruit gaussien