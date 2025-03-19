# Système de détection automatique des carotides

## Aperçu du projet

Ce projet développe un système avancé d'intelligence artificielle pour la détection automatique des artères carotides dans les images médicales DICOM. Les carotides, artères principales qui alimentent le cerveau en sang, sont essentielles à surveiller pour la prévention des accidents vasculaires cérébraux (AVC) et autres complications cardiovasculaires. 

Notre solution répond à un besoin clinique crucial : l'analyse manuelle des carotides sur les images scanner est un processus chronophage, sujet à la variabilité inter-observateur et nécessitant une expertise spécialisée. Notre système automatise cette tâche en exploitant les dernières avancées en deep learning, spécifiquement l'architecture YOLOv8 (You Only Look Once), permettant une détection précise, rapide et reproductible.

Le pipeline complet comprend le prétraitement avancé des images DICOM, un système d'annotation assistée, l'entraînement optimisé d'un modèle de détection et un module d'inférence adaptable pour l'analyse de nouvelles images. Cette solution s'intègre parfaitement aux workflows cliniques existants et peut servir d'assistant au diagnostic pour les radiologues et neurologues.

## Table des matières
- [Fondements techniques](#fondements-techniques)
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
- [Performances et validation](#performances-et-validation)
- [Applications cliniques](#applications-cliniques)
- [Dépannage](#dépannage)
- [Développements futurs](#développements-futurs)
- [Licence](#licence)

## Fondements techniques

Le système repose sur plusieurs innovations techniques clés :

### Traitement d'images médicales avancé
Notre prétraitement utilise des techniques sophistiquées d'amélioration d'image spécifiquement optimisées pour les structures vasculaires :
- **Double application de CLAHE** (Contrast Limited Adaptive Histogram Equalization) : Révèle les contours subtils des carotides en améliorant le contraste local tout en limitant l'amplification du bruit
- **Correction gamma adaptative** : Accentue les zones d'intérêt clinique tout en préservant les nuances diagnostiques essentielles
- **Filtrage bilatéral paramétré** : Réduit le bruit inhérent aux images scanner tout en préservant les frontières anatomiques critiques
- **Système de grille d'annotation** : Facilite le positionnement précis et reproductible des annotations, améliorant la qualité du jeu de données d'entraînement

### Architecture de deep learning spécialisée
Nous utilisons une adaptation personnalisée de YOLOv8, comprenant :
- Un **backbone CSPDarknet** optimisé pour l'extraction de caractéristiques pertinentes aux structures vasculaires
- Une **architecture de cou PANet** pour la fusion efficace d'informations multi-échelles, cruciale pour détecter les carotides de différentes tailles
- Des **têtes de détection découplées** permettant une localisation précise et une classification fiable des carotides gauche et droite

### Algorithme d'inférence adaptatif
Notre système d'inférence utilise une approche multi-seuils innovante qui :
- Applique progressivement différents niveaux de confiance (0.25, 0.15, 0.1)
- S'arrête automatiquement lorsque les deux carotides sont détectées
- Adapte sa sensibilité aux caractéristiques spécifiques de chaque image
- Optimise le compromis entre précision et temps de traitement

## Prérequis

- Python 3.8+ (compatible avec Python 3.10)
- CUDA 11.2+ (fortement recommandé pour l'entraînement et l'inférence accélérés)
- RAM: 8GB minimum, 16GB recommandé pour le traitement par lots
- GPU: NVIDIA avec au moins 6GB VRAM pour l'entraînement optimal

### Bibliothèques Python essentielles:
  - pydicom 2.3+ : Manipulation des fichiers DICOM médicaux
  - numpy 1.20+ : Opérations numériques et manipulation de matrices
  - opencv-python 4.5+ : Traitement d'image avancé
  - Pillow 8.0+ : Opérations supplémentaires sur les images
  - ultralytics 8.0+ : Framework YOLOv8
  - pandas 1.3+ : Analyse et manipulation des données d'annotation
  - PyYAML 6.0+ : Gestion des configurations

## Structure du projet

```
IADetectionCarotide/
│
├── data/                         # Dossier de données structuré
│   ├── images/                   # Images prétraitées pour entraînement/validation/test
│   │   ├── train/               # Sous-ensemble d'entraînement (70%)
│   │   ├── val/                 # Sous-ensemble de validation (15%)
│   │   └── test/                # Sous-ensemble de test (15%)
│   ├── preview/                  # Images avec grille pour faciliter l'annotation
│   ├── labels/                   # Annotations converties au format YOLO
│   │   ├── train/               # Étiquettes pour les images d'entraînement
│   │   ├── val/                 # Étiquettes pour les images de validation
│   │   └── test/                # Étiquettes pour les images de test
│   ├── annotations/              # Annotations brutes (format VIA)
│   ├── failed/                   # Images problématiques pour analyse et débogage
│   └── classes.txt               # Définition des classes (carotide gauche/droite)
│
├── carotid_detection/            # Résultats d'entraînement et modèles
│   └── train1/                   # Session d'entraînement spécifique
│       ├── weights/              # Poids du modèle entraîné
│       │   ├── best.pt           # Meilleur modèle (mAP le plus élevé)
│       │   └── last.pt           # Dernier point de contrôle
│       ├── results.csv           # Métriques d'entraînement détaillées 
│       └── args.yaml             # Configuration utilisée pour l'entraînement
│
├── resultats_detection/          # Résultats d'inférence et analyses
│   └── detection_YYYYMMDD_HHMMSS/ # Regroupement par session de détection
│       ├── *.png                 # Images annotées avec détections
│       ├── detection_log.txt     # Journal des opérations de détection
│       └── detection_summary.txt # Résumé analytique des résultats
│
├── process_dicom.py              # Module de prétraitement des images DICOM
├── convert_annotations.py        # Convertisseur d'annotations VIA vers YOLO
├── train_yolo.py                 # Script d'entraînement du modèle YOLOv8
├── detect_carotide.py            # Système d'inférence et d'analyse
├── data.yaml                     # Configuration du dataset pour YOLOv8
└── requirements.txt              # Dépendances précises avec versions
```

## Installation

1. Cloner le dépôt:
   ```bash
   git clone https://github.com/uncyzer/TPI-RQCL-DeepBridge.git
   cd IADetectionCarotide
   ```

2. Créer un environnement virtuel isolé et l'activer:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. Installer les dépendances avec versions spécifiques:
   ```bash
   pip install -r requirements.txt
   ```

4. Vérifier l'installation CUDA (pour l'accélération GPU):
   ```bash
   python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('Version CUDA:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
   ```

## Utilisation

### 1. Prétraitement des images DICOM

Le module de prétraitement transforme les images DICOM brutes en entrées optimisées pour la détection des carotides, en appliquant des techniques d'amélioration d'image spécifiquement paramétrées pour les structures vasculaires.

Pour traiter un dossier d'images DICOM et les préparer pour l'annotation:

```bash
python process_dicom.py /chemin/vers/images/dicom --output-dir carotid_project/data --darkness 0.5
```

#### Options avancées:
- `input_path`: Chemin vers le dossier contenant les fichiers DICOM (requis)
- `--output-dir`: Dossier de sortie où seront stockées les images traitées (par défaut: 'carotid_project/data')
- `--darkness`: Facteur d'assombrissement, entre 0 et 1, permettant de révéler les structures vasculaires par rapport au fond (par défaut: 0.5)

#### Processus de prétraitement:
1. **Validation des fichiers DICOM** - Vérification de l'intégrité et de la compatibilité des images
2. **Normalisation adaptative** - Ajustement de la plage dynamique basé sur les percentiles pour optimiser le contraste
3. **Amélioration du contraste** - Application de CLAHE avec paramètres optimisés pour les structures vasculaires
4. **Correction gamma** - Ajustement non-linéaire pour améliorer la visibilité des zones d'intérêt
5. **Réduction du bruit** - Filtrage bilatéral préservant les bords des structures anatomiques
6. **Création de la grille d'annotation** - Superposition d'une grille paramétrable facilitant le positionnement précis des annotations

#### Résultat:
- Images prétraitées de haute qualité dans `/output-dir/images`
- Images avec grille d'aide à l'annotation dans `/output-dir/preview`
- Copies des fichiers problématiques dans `/output-dir/failed` pour analyse ultérieure

### 2. Annotation des images

Le processus d'annotation utilise VGG Image Annotator (VIA), un outil open-source puissant spécialement adapté à l'annotation médicale:

1. Ouvrir VIA dans votre navigateur (https://www.robots.ox.ac.uk/~vgg/software/via/)
2. Charger les images avec grille depuis le dossier `carotid_project/data/preview`
3. Configurer l'attribut de région:
   - Nom: "carotide"
   - Type: Options de sélection
   - Valeurs: "carotide gauche", "carotide droite"
4. Annoter les carotides selon les règles suivantes:
   - Utiliser des rectangles englobants précis autour des carotides
   - S'assurer que chaque image contient les deux carotides annotées
   - Utiliser la grille comme référence pour un positionnement cohérent
   - Veiller à distinguer correctement carotide gauche et droite
5. Exporter les annotations au format CSV dans `carotid_project/data/annotations/via_project_csv.csv`

### 3. Conversion des annotations

Le script `convert_annotations.py` transforme les annotations VIA au format requis par YOLOv8 et organise le jeu de données en ensembles d'entraînement, validation et test:

```bash
python convert_annotations.py
```

#### Processus de conversion:
1. **Lecture des annotations VIA** - Analyse du fichier CSV exporté
2. **Normalisation des coordonnées** - Conversion des coordonnées absolues (pixels) en valeurs relatives requises par YOLO
3. **Conversion des classes** - Mappage des libellés textuels aux indices numériques de classe
4. **Division stratifiée des données** - Répartition des images en ensembles d'entraînement (70%), validation (15%) et test (15%)
5. **Génération des fichiers d'étiquettes** - Création d'un fichier .txt par image au format YOLO
6. **Configuration du dataset** - Création du fichier data.yaml définissant les classes et chemins pour l'entraînement

### 4. Entraînement du modèle

Le script `train_yolo.py` configure et exécute l'entraînement d'un modèle YOLOv8 personnalisé pour la détection des carotides:

```bash
python train_yolo.py
```

#### Processus d'entraînement:
1. **Initialisation du modèle** - Chargement d'un modèle YOLOv8 nano pré-entraîné comme point de départ
2. **Configuration de l'entraînement** - Paramétrage des hyperparamètres optimisés pour la détection des carotides
3. **Augmentation de données** - Application automatique de transformations pour améliorer la robustesse du modèle
4. **Entraînement itératif** - Apprentissage progressif sur 100 époques avec arrêt anticipé si nécessaire
5. **Validation continue** - Évaluation des performances sur l'ensemble de validation à chaque époque
6. **Sauvegarde des points de contrôle** - Conservation du meilleur modèle selon la métrique mAP50-95

#### Paramètres avancés:
Vous pouvez modifier les paramètres d'entraînement en éditant directement le script `train_yolo.py`:
- Nombre d'époques (défaut: 100)
- Taille des images (défaut: 640x640)
- Taille du batch (défaut: 16, à ajuster selon la capacité de votre GPU)
- Seuil de patience pour l'arrêt anticipé (défaut: 20 époques)
- Périphérique de calcul (GPU ou CPU)

### 5. Détection des carotides

Le script `detect_carotide.py` implémente un système d'inférence adaptative et analyse des nouvelles images DICOM:

```bash
python detect_carotide.py
```

#### Configuration préalable:
Avant l'exécution, modifiez les chemins dans le script selon votre environnement:
- `BASE_DIR`: Dossier contenant les nouvelles images DICOM à analyser
- `MODEL_PATH`: Chemin vers le modèle entraîné (généralement `carotid_detection/train*/weights/best.pt`)
- `OUTPUT_DIR`: Dossier où sauvegarder les résultats de détection

#### Processus de détection:
1. **Lecture et prétraitement** - Chargement des images DICOM et application du pipeline d'amélioration d'image
2. **Détection adaptive** - Application progressive de seuils de confiance décroissants pour maximiser la détection des carotides
3. **Filtrage des résultats** - Sélection des meilleures détections selon les critères de confiance et de position
4. **Génération de visualisations** - Création d'images annotées montrant les carotides détectées
5. **Analyse quantitative** - Extraction de métriques morphologiques et de confiance pour chaque détection
6. **Production de rapports** - Génération d'un rapport détaillé résumant les résultats de l'analyse

#### Résultats générés:
- Images annotées dans `OUTPUT_DIR` montrant les carotides détectées
- Journal de détection `detection_log.txt` documentant le processus
- Rapport d'analyse `detection_summary.txt` présentant les résultats de manière structurée

## Paramètres et configuration

### Paramètres de prétraitement (`ProcessingConfig` dans `process_dicom.py`)

Ces paramètres contrôlent le comportement du pipeline de prétraitement des images, chacun ayant un impact spécifique sur le résultat:

- `clahe_clip_limit` (défaut: 3.0): Limite de contraste pour l'algorithme CLAHE
  - Valeurs plus élevées: Augmentation du contraste mais potentiellement plus de bruit
  - Valeurs plus basses: Contraste plus subtil mais préservation des textures fines
  
- `clahe_grid_size` (défaut: (8, 8)): Taille de la grille pour l'algorithme CLAHE
  - Grille plus fine (ex: 16x16): Meilleure adaptation aux détails locaux
  - Grille plus grossière: Meilleure cohérence globale du contraste
  
- `bilateral_d` (défaut: 7): Diamètre du filtre bilatéral
  - Valeurs plus élevées: Lissage plus important mais temps de traitement accru
  - Valeurs plus basses: Préservation des détails fins mais moins de réduction du bruit
  
- `bilateral_sigma_color` (défaut: 30): Sigma couleur pour le filtre bilatéral
  - Paramètre contrôlant la tolérance aux variations de contraste
  
- `bilateral_sigma_space` (défaut: 30): Sigma espace pour le filtre bilatéral
  - Paramètre contrôlant l'influence de la distance spatiale
  
- `darkness_factor` (défaut: 0.7): Facteur d'assombrissement
  - 1.0: Aucun assombrissement
  - 0.5: Réduction de 50% de la luminosité pour faire ressortir les structures vasculaires
  
- `grid_divisions` (défaut: 12): Nombre de divisions dans la grille d'annotation
  - Valeurs plus élevées: Grille plus fine pour un positionnement plus précis
  - Valeurs plus basses: Grille moins intrusive visuellement

### Configuration optimale validée cliniquement

Après des tests exhaustifs sur différentes séries d'images, nous recommandons la configuration suivante qui offre un équilibre optimal entre rehaussement des carotides et préservation des détails anatomiques:

```python
config = ProcessingConfig(
    clahe_clip_limit=2.0,  
    clahe_grid_size=(16, 16),  
    bilateral_d=9,  
    bilateral_sigma_color=30,
    bilateral_sigma_space=30,
    darkness_factor=0.7,
    grid_divisions=12
)
```

### Paramètres d'entraînement YOLOv8

Les paramètres d'entraînement influencent directement les performances du modèle et doivent être ajustés en fonction de votre jeu de données:

- `epochs` (défaut: 100): Nombre de passages complets sur le jeu de données
  - Augmentez pour les jeux de données plus complexes ou plus volumineux
  - L'arrêt anticipé empêche le surapprentissage si la performance stagne
  
- `imgsz` (défaut: 640): Taille des images d'entrée
  - Valeurs plus élevées: Meilleure détection des petits objets mais consommation mémoire accrue
  - 640 représente un bon compromis pour la détection des carotides
  
- `batch` (défaut: 16): Nombre d'images traitées simultanément
  - Ajustez en fonction de la capacité de votre GPU (réduisez pour les GPUs avec moins de VRAM)
  
- `patience` (défaut: 20): Nombre d'époques sans amélioration avant arrêt anticipé
  - Valeurs plus élevées: Entraînement plus long avec possibilité de surmonter les plateaux
  - Valeurs plus basses: Temps d'entraînement plus court
  
- `device` (défaut: auto): Périphérique de calcul ('0' pour première GPU, 'cpu' pour CPU)
  - L'entraînement sur GPU est fortement recommandé (10-50x plus rapide)

## Applications cliniques

Ce système de détection automatique des carotides offre plusieurs applications pratiques dans le contexte clinique:

### Assistance au diagnostic
- Détection rapide et systématique des carotides sur les examens scanner
- Localisation précise pour faciliter l'évaluation des sténoses et autres pathologies
- Réduction du temps d'analyse pour les radiologues et neurologues

### Recherche et études cliniques
- Analyse standardisée de grandes séries d'images pour les études épidémiologiques
- Extraction automatisée de cohortes basées sur des critères anatomiques spécifiques
- Quantification objective des changements morphologiques dans les études longitudinales

### Intégration aux workflows existants
- Compatible avec les systèmes PACS (Picture Archiving and Communication System)
- Peut fonctionner comme module de pré-analyse avant revue par le radiologue
- Extensible pour inclure d'autres structures anatomiques d'intérêt

## Dépannage

### Images DICOM non reconnues
- **Problème**: Certains fichiers DICOM ne sont pas correctement traités
- **Solutions**:
  - Vérifiez la conformité des fichiers au standard DICOM
  - Utilisez l'option `force=True` dans `pydicom.dcmread()` pour les formats non standard
  - Examinez les fichiers dans le dossier `failed/` pour identifier les problèmes spécifiques
  - Pour les fichiers DICOMDIR, utilisez des outils spécialisés pour extraire les images individuelles

### Problèmes de chemin sur Windows
- **Problème**: Erreurs liées aux chemins trop longs sur Windows
- **Solution**:
  - Le script utilise automatiquement le préfixe '\\\\?\\' pour contourner la limitation de 260 caractères
  - Évitez les structures de dossiers profondément imbriquées
  - Placez le projet près de la racine du disque pour raccourcir les chemins

### Erreurs d'entraînement
- **Problème**: L'entraînement échoue ou donne des résultats médiocres
- **Solutions**:
  - Vérifiez que le fichier `data.yaml` pointe vers les bons chemins absolus
  - Assurez-vous que les classes sont correctement définies et cohérentes
  - Pour les systèmes avec peu de mémoire GPU:
    - Réduisez la taille du batch (8, 4 ou même 2)
    - Diminuez la taille des images d'entrée (416 au lieu de 640)
  - Examinez attentivement les courbes d'apprentissage pour détecter surapprentissage ou sous-apprentissage

### Détections incorrectes
- **Problème**: Faux positifs ou faux négatifs lors de la détection
- **Solutions**:
  - Ajustez les seuils de confiance dans `detect_carotide.py`
  - Augmentez la diversité du jeu de données d'entraînement
  - Vérifiez la qualité des annotations d'entraînement
  - Pour les anatomies atypiques, envisagez un fine-tuning spécifique du modèle

## Développements futurs

Notre feuille de route pour l'évolution du système comprend:

### Améliorations techniques prévues
- Support multi-GPU pour l'entraînement distribué
- Implémentation d'un mécanisme d'apprentissage continu
- Optimisation pour déploiement sur appareils à ressources limitées
- Intégration d'une interface utilisateur web pour faciliter l'utilisation

### Extensions fonctionnelles
- Détection d'autres structures vasculaires (artères vertébrales, tronc artériel)
- Classification automatique des sténoses carotidiennes
- Mesure automatique des diamètres et calcul du pourcentage de sténose
- Analyse de la composition des plaques d'athérome

### Intégration aux systèmes cliniques
- Développement de connecteurs DICOM pour intégration PACS
- Implémentation de rapports structurés selon les standards radiologiques
- Création d'une API REST pour intégration avec des systèmes tiers
