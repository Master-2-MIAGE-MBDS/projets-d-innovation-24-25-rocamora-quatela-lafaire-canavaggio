# Système de détection automatique des carotides

Ce projet implémente un système complet pour la détection automatique des carotides dans des images médicales DICOM, en utilisant un réseau de neurones YOLOv8. Le pipeline comprend le prétraitement des images DICOM, l'annotation manuelle assistée, l'entraînement d'un modèle de détection et l'inférence sur de nouvelles images.

## Table des matières
- [Prérequis](#prérequis)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
  - [1. Prétraitement des images DICOM](#1-prétraitement-des-images-dicom)
  - [2. Annotation des images](#2-annotation-des-images)
  - [3. Conversion des annotations](#3-conversion-des-annotations)
  - [4. Entraînement du modèle](#4-entraînement-du-modèle)
  - [5. Détection des carotides](#5-détection-des-carotides)
- [Paramètres et configuration](#paramètres-et-configuration)
- [Performances](#performances)
- [Dépannage](#dépannage)
- [Licence](#licence)

## Prérequis

- Python 3.8+
- CUDA (recommandé pour l'entraînement)
- Bibliothèques Python:
  - pydicom
  - numpy
  - opencv-python
  - Pillow
  - ultralytics
  - pandas
  - PyYAML

## Structure du projet

```
tpi/
│
├── data/                         # Dossier de données
│   ├── images/                   # Images prétraitées
│   ├── preview/                  # Images avec grille pour annotation
│   ├── labels/                   # Annotations au format YOLO
│   ├── annotations/              # Annotations brutes (format VIA)
│   ├── failed/                   # Images problématiques
│   └── classes.txt               # Définition des classes
│
├── carotid_detection/            # Dossier des résultats d'entraînement
│   └── train1/                   # Résultats d'un entraînement spécifique
│       └── weights/
│           └── best.pt           # Meilleur modèle entraîné
│
├── resultats_detection/          # Résultats de détection
│
├── process_dicom.py              # Script de prétraitement des images DICOM
├── convert_annotations.py        # Script de conversion des annotations
├── train_yolo.py                 # Script d'entraînement du modèle
├── detect_carotide.py            # Script de détection sur nouvelles images
├── data.yaml                     # Configuration des données pour YOLO
└── requirements.txt              # Fichier de configuration des dépendances
```

## Installation

1. Cloner le dépôt:
   ```bash
   git clone https://github.com/dragun06/tpi.git
   cd tpi
   ```

2. Créer un environnement virtuel et l'activer:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. Installer les dépendances:
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

### 1. Prétraitement des images DICOM

Pour traiter un dossier d'images DICOM et les préparer pour l'annotation:

```bash
python process_dicom.py /chemin/vers/images/dicom --output-dir carotid_project/data --darkness 0.5
```

Options:
- `input_path`: Chemin vers le dossier contenant les fichiers DICOM
- `--output-dir`: Dossier de sortie (par défaut: 'carotid_project/data')
- `--darkness`: Facteur d'assombrissement, entre 0 et 1 (par défaut: 0.5)

Résultat:
- Images prétraitées dans `/output-dir/images`
- Images avec grille pour faciliter l'annotation dans `/output-dir/preview`

### 2. Annotation des images

Utilisez un outil d'annotation comme VGG Image Annotator (VIA) pour annoter les carotides:

1. Ouvrir VIA (https://www.robots.ox.ac.uk/~vgg/software/via/)
2. Charger les images depuis le dossier `carotid_project/data/preview`
3. Créer un attribut de région nommé "carotide" avec des valeurs "carotide gauche" et "carotide droite"
4. Annoter les carotides avec des rectangles
5. Exporter les annotations au format CSV dans `carotid_project/data/annotations/via_project_csv.csv`

### 3. Conversion des annotations

Pour convertir les annotations VIA au format YOLO:

```bash
python convert_annotations.py
```

Ce script:
- Lit les annotations VIA
- Les convertit au format YOLO
- Divise les données en ensembles d'entraînement, validation et test
- Crée le fichier data.yaml pour l'entraînement

### 4. Entraînement du modèle

Pour entraîner un modèle YOLOv8 sur les données annotées:

```bash
python train_yolo.py
```

Ce script:
- Charge un modèle YOLOv8 nano pré-entraîné
- L'entraîne sur les données annotées des carotides
- Sauvegarde le modèle et les résultats dans le dossier `carotid_detection/`

Vous pouvez modifier les paramètres d'entraînement (comme le nombre d'époques, la taille du batch) directement dans le script.

### 5. Détection des carotides

Pour détecter les carotides dans de nouvelles images DICOM:

```bash
python detect_carotide.py
```

Avant l'exécution, assurez-vous de modifier les chemins dans le script:
- `BASE_DIR`: Dossier contenant les nouvelles images DICOM
- `MODEL_PATH`: Chemin vers le modèle entraîné
- `OUTPUT_DIR`: Dossier où sauvegarder les résultats

Le script:
- Traite les images DICOM
- Applique le modèle pour détecter les carotides
- Sauvegarde les images avec détections
- Génère un rapport détaillé des résultats

## Paramètres et configuration

### Paramètres de prétraitement (`ProcessingConfig` dans `process_dicom.py`)

- `clahe_clip_limit`: Limite de contraste pour CLAHE (par défaut: 3.0)
- `clahe_grid_size`: Taille de la grille pour CLAHE (par défaut: (8, 8))
- `bilateral_d`: Diamètre du filtre bilatéral (par défaut: 7)
- `bilateral_sigma_color`: Sigma couleur pour le filtre bilatéral (par défaut: 30)
- `bilateral_sigma_space`: Sigma espace pour le filtre bilatéral (par défaut: 30)
- `darkness_factor`: Facteur d'assombrissement (par défaut: 0.7)
- `grid_divisions`: Nombre de divisions dans la grille d'annotation (par défaut: 12)

### Paramètres d'entraînement YOLOv8

Configurations disponibles dans `train_yolo.py`:
- `epochs`: Nombre d'époques d'entraînement (par défaut: 100)
- `imgsz`: Taille des images d'entrée (par défaut: 640)
- `batch`: Taille du batch (par défaut: 16)
- `patience`: Patience pour l'arrêt anticipé (par défaut: 20)
- `device`: Appareil pour l'entraînement ('0' pour GPU, 'cpu' pour CPU)

## Performances

Le modèle entraîné atteint les performances suivantes sur l'ensemble de validation:
- Précision: ~99%
- Rappel: 100%
- mAP50: 99.5%
- mAP50-95: ~53%

Ces métriques indiquent une bonne performance du modèle pour la détection des carotides.

## Dépannage

### Images DICOM non reconnues
- Vérifiez que les fichiers sont bien au format DICOM
- Utilisez l'option `force=True` dans `pydicom.dcmread()` pour les formats non standard
- Les fichiers problématiques sont copiés dans le dossier `failed/`

### Problèmes de chemin sur Windows
Pour les chemins longs sur Windows, le script utilise automatiquement le préfixe '\\\\?\\'.

### Erreurs d'entraînement
- Vérifiez que le fichier `data.yaml` pointe vers les bons chemins
- Assurez-vous que les classes sont correctement définies
- Pour les systèmes avec peu de mémoire GPU, réduisez la taille du batch
