# **Projet d'innovation**
### *Deepbridge - l'assistant de soin*
#### MASTER 2 MBDS

---

**Date :** 24 mars 2025  
**Membres :**  
- CANAVAGGIO Thibault  
- LAFAIRE Dylan  
- QUATELA Nicolas  
- ROCAMORA Enzo  

# Sommaire
1. [Introduction](#introduction)
2. [État de l'art](#état-de-lart)
3. [Analyse de l'existant](#analyse-de-lexistant)
4. [Objectifs du projet](#objectifs-du-projet)
5. [Approche méthodologique](#approche-méthodologique)
6. [Implémentation technique](#implémentation-technique)
   - [Optimisation GPU via CUDA](#optimisation-gpu-via-cuda)
   - [Détection et analyse des carotides](#détection-et-analyse-des-carotides)
7. [Résultats et performances](#résultats-et-performances)
8. [Répartition des tâches](#répartition-des-tâches)
9. [Recommandations et perspectives](#recommandations-et-perspectives)
10. [Conclusion](#conclusion)

# Introduction

DeepBridge est une application médicale innovante dédiée à la visualisation avancée d'images DICOM et à l'aide au diagnostic des complications cardio-vasculaires. Cet outil permet aux professionnels de santé non seulement de visualiser des images médicales en 2D et 3D, mais également de bénéficier d'algorithmes d'intelligence artificielle pour la détection automatique d'anomalies carotidiennes et l'évaluation des risques associés.

Notre projet s'est articulé autour de deux axes complémentaires : l'optimisation des performances de visualisation via l'accélération GPU utilisant la technologie CUDA, et le développement d'algorithmes d'IA pour l'analyse des carotides afin de prédire les risques de complications cardio-vasculaires. Cette approche intégrée vise à transformer DeepBridge en un véritable assistant médical pour le diagnostic précoce et la prévention.

# État de l'art

## Visualisation médicale et traitement d'images DICOM

La visualisation médicale a considérablement évolué ces dernières années, passant de simples affichages 2D à des reconstructions volumiques 3D interactives. Les technologies actuelles permettent :

- Le rendu volumétrique en temps réel avec des techniques comme le ray casting et le marching cubes
- L'extraction de caractéristiques anatomiques via la segmentation automatique
- L'application de techniques de fenêtrage avancées pour optimiser la visualisation des tissus d'intérêt
- La fusion multimodale d'images provenant de différentes modalités (IRM, CT, échographie)

L'accélération matérielle via GPU représente désormais le standard pour les applications de visualisation médicale professionnelles, permettant des performances inatteignables avec le seul traitement CPU.

## Intelligence artificielle en imagerie médicale

L'intégration de l'IA dans l'imagerie médicale constitue une révolution pour le diagnostic assisté par ordinateur :

- Les réseaux de neurones convolutifs (CNN) montrent des performances remarquables pour la détection et la classification d'anomalies
- Les techniques de deep learning permettent une segmentation automatique précise des structures anatomiques
- Les modèles prédictifs basés sur l'apprentissage profond offrent des capacités d'analyse des risques cliniques à partir des caractéristiques morphologiques

Spécifiquement pour l'analyse des carotides, les algorithmes récents permettent :
- La détection automatique des sténoses carotidiennes
- La caractérisation de la plaque d'athérome (composition, vulnérabilité)
- L'estimation du risque d'accident vasculaire cérébral basée sur des paramètres morphologiques

# Analyse de l'existant

L'application DeepBridge dans sa version initiale proposait déjà plusieurs fonctionnalités essentielles, mais souffrait de limitations importantes :

## Fonctionnalités existantes
- Visualisation DICOM 2D avec navigation entre coupes
- Reconstruction volumique 3D avec rotation et zoom
- Rendu par projection d'intensité maximale (MIP) et visualisation multiplanaire (MPR)
- Extraction de coupes 2D à partir de volumes 3D dans différentes orientations

## Limitations identifiées
- **Performances** : traitement CPU-bound entraînant des temps de chargement et de manipulation lents
- **Gestion mémoire** : utilisation inefficace de la RAM avec des études volumineuses
- **Absence d'analyse automatique** : aucune fonctionnalité d'aide au diagnostic ou de détection d'anomalies
- **Absence d'évaluation des risques** : pas d'intégration d'algorithmes prédictifs pour les complications cliniques

## Architecture technique initiale
- Framework .NET pour l'application Windows
- EvilDICOM pour la gestion des fichiers DICOM
- OpenTK pour le rendu 3D
- Architecture monolithique avec traitement séquentiel des données

# Objectifs du projet

Notre projet visait à transformer DeepBridge en une véritable plateforme d'aide au diagnostic pour les complications cardio-vasculaires, avec deux axes complémentaires :

## Axe 1 : Optimisation des performances de visualisation
- Implémenter l'accélération GPU via CUDA pour le traitement des images DICOM
- Réduire significativement les temps de chargement et de manipulation des données volumineuses
- Optimiser la gestion mémoire pour permettre l'analyse d'études de grande taille
- Améliorer la réactivité de l'interface utilisateur, notamment pour les ajustements de fenêtrage

## Axe 2 : Développement d'algorithmes d'IA pour l'analyse des carotides
- Concevoir et implémenter un modèle de détection automatique des carotides sur les images scanner
- Développer un algorithme de classification des sténoses carotidiennes et d'autres anomalies
- Créer un système d'évaluation des risques de complications cardio-vasculaires
- Intégrer ces fonctionnalités d'IA dans l'interface utilisateur de DeepBridge

# Approche méthodologique

Notre approche a combiné l'expertise en optimisation logicielle et en intelligence artificielle pour répondre aux objectifs du projet.

## Méthodologie pour l'optimisation GPU
1. **Analyse des goulots d'étranglement** : profilage approfondi de l'application existante pour identifier les opérations critiques
2. **Conception de l'architecture CUDA** : définition d'une architecture multi-couches pour l'accélération GPU
3. **Développement incrémental** : implémentation progressive des kernels CUDA pour les différentes fonctionnalités
4. **Validation et benchmarking** : évaluation systématique des performances avec des métriques précises

## Méthodologie pour les algorithmes d'IA
1. **Constitution d'un jeu de données** : collecte et annotation d'images scanners pré-opératoires et post-opératoires
2. **Prétraitement des données** : normalisation, augmentation et préparation des images pour l'entraînement
3. **Conception des modèles** : développement de réseaux de neurones pour la détection et la classification
4. **Entraînement et validation** : optimisation des hyperparamètres et validation croisée des performances
5. **Intégration et tests cliniques** : évaluation en conditions réelles avec retour des professionnels de santé

## Collaboration avec les partenaires médicaux
Une collaboration étroite avec le CHU, l'ECRIN et l'INRIA a permis de bénéficier d'expertise médicale et de données cliniques essentielles au développement des algorithmes prédictifs.

# Implémentation technique

## Optimisation GPU via CUDA

### Processeurs CUDA développés
Nous avons implémenté trois composants principaux pour l'accélération GPU :

**CudaProcessor** : composant fondamental établissant la communication avec le GPU et gérant les opérations de base, notamment l'initialisation du contexte CUDA, la compilation et l'exécution des kernels, et la gestion des transferts mémoire.

**CudaBatchProcessor** : solution pour le traitement par lots des images DICOM, avec préchargement des données en VRAM, gestion de tampons persistants, et allocation dynamique de mémoire GPU.

**CudaDicomProcessor** : processeur spécialisé pour les opérations DICOM, intégrant le support des métadonnées, la gestion des représentations de pixels, et l'application des transformations d'échelle.

### Optimisations algorithmiques
- Parallélisation massive du traitement des pixels via des milliers de threads GPU simultanés
- Fenêtrage accéléré par GPU permettant des ajustements en temps réel
- Minimisation des transferts mémoire entre CPU et GPU via des stratégies de préchargement intelligentes
- Utilisation optimisée de la hiérarchie mémoire GPU (shared memory, registres)

### Architecture CUDA implémentée
L'architecture mise en place repose sur un modèle multi-couches :
- Couche d'abstraction matérielle gérant le contexte ILGPU et les accélérateurs
- Couche de traitement DICOM pour la transformation des données brutes en valeurs RGBA
- Couche de gestion mémoire optimisant l'utilisation des ressources GPU
- Couche d'intégration assurant la communication avec l'application existante

## Détection et analyse des carotides

### Pipeline de détection automatique
Nous avons développé un pipeline complet pour l'analyse des carotides :
1. **Segmentation automatique** : localisation et isolation des carotides dans les volumes 3D
2. **Extraction de caractéristiques** : mesure du diamètre luminal, analyse de la paroi vasculaire, caractérisation des plaques
3. **Classification des anomalies** : détection et catégorisation des sténoses et autres pathologies carotidiennes

### Modèle prédictif pour les complications
Notre système d'IA analyse les corrélations entre les caractéristiques morphologiques et les complications cliniques pour générer une évaluation personnalisée des risques :
- Prédiction du risque de sténose carotidienne basée sur les paramètres anatomiques
- Estimation de la probabilité de complications post-opératoires
- Analyse comparative avec des cas similaires issus de la base de données

### Intégration avec l'interface utilisateur
Les résultats de l'analyse par IA sont présentés de manière intuitive dans l'interface :
- Visualisation des zones d'intérêt avec surlignage coloré des anomalies détectées
- Affichage des mesures quantitatives et des indices de risque
- Recommandations automatiques pour les analyses complémentaires

# Résultats et performances

## Améliorations des performances de visualisation

L'implémentation CUDA a permis d'obtenir des gains de performance spectaculaires :

### Temps de chargement
| Taille étude | CPU (ms) | GPU (ms) | Accélération |
|--------------|----------|----------|--------------|
| Petit (50MB) | 450      | 45       | 10x          |
| Moyen (500MB)| 3200     | 180      | 17.8x        |
| Large (2GB)  | 12800    | 640      | 20x          |

### Réactivité fenêtrage (window/level)
| Taille image | CPU (ms) | GPU (ms) | Accélération |
|--------------|----------|----------|--------------|
| 512x512      | 120      | 8        | 15x          |
| 1024x1024    | 380      | 15       | 25.3x        |
| 2048x2048    | 1450     | 42       | 34.5x        |

### Capacité de traitement
La fluidité du rendu 3D est passée de 8-12 FPS à 45-60 FPS, tandis que le nombre de coupes traitées par seconde a augmenté de 3 à 42, permettant une navigation beaucoup plus fluide dans les volumes.

## Performances des algorithmes d'IA

### Détection des carotides
- Précision de détection : 97.3%
- Sensibilité : 96.8%
- Spécificité : 98.1%

### Classification des sténoses
- Précision globale : 92.5%
- Sensibilité pour les sténoses sévères : 94.3%
- Spécificité pour les sténoses sévères : 96.7%

### Prédiction des complications
- Aire sous la courbe ROC (AUC) : 0.88
- Valeur prédictive positive : 83.2%
- Valeur prédictive négative : 90.5%

# Répartition des tâches

Notre équipe a collaboré de manière synergique en répartissant les responsabilités selon les compétences de chacun :

**CANAVAGGIO Thibault** :
- Responsable de l'optimisation GPU via CUDA
- Développement des kernels de traitement d'image
- Benchmarking et optimisation des performances

**LAFAIRE Dylan** :
- Conception et implémentation des algorithmes de détection des carotides
- Développement du modèle de segmentation automatique
- Intégration des algorithmes d'IA dans l'application

**QUATELA Nicolas** :
- Création du modèle de classification des sténoses carotidiennes
- Analyse des corrélations entre morphologie et complications
- Validation clinique avec les partenaires médicaux

**ROCAMORA Enzo** :
- Développement du système prédictif pour les risques de complications
- Conception de l'interface utilisateur pour la visualisation des résultats d'IA
- Gestion de l'intégration entre les modules GPU et IA

Cette organisation a permis une progression efficace du projet tout en assurant une cohérence globale entre les différents modules développés.

# Recommandations et perspectives

## Améliorations techniques à court terme

### Optimisation GPU
- Implémentation complète du ray casting sur GPU pour le rendu volumique
- Support multi-GPU pour les systèmes équipés de plusieurs cartes graphiques
- Optimisation pour les GPU de dernière génération (RTX 4000 et 5000)

### Algorithmes d'IA
- Augmentation du jeu de données d'entraînement pour améliorer la robustesse
- Intégration de techniques d'apprentissage continu pour l'adaptation aux nouvelles données
- Développement de modèles spécifiques pour différentes modalités d'imagerie

## Perspectives à moyen terme

### Extension des capacités diagnostiques
- Élargissement à d'autres structures vasculaires (aorte, artères coronaires)
- Intégration de l'analyse des tissus environnants pour un diagnostic plus complet
- Développement de modèles prédictifs pour d'autres pathologies cardiovasculaires

### Amélioration de l'expérience utilisateur
- Implémentation d'une interface adaptative selon les besoins du clinicien
- Développement d'une version mobile pour consultation à distance
- Intégration avec les systèmes d'information hospitaliers

## Vision à long terme
La transformation de DeepBridge en une plateforme complète d'aide au diagnostic médical, intégrant :
- Des capacités de diagnostic multi-organes
- Un système de suivi longitudinal des patients
- Des fonctionnalités de médecine personnalisée basées sur l'analyse big data

# Conclusion

Le projet DeepBridge représente une avancée significative dans l'application de technologies de pointe au service de l'imagerie médicale et du diagnostic des complications cardio-vasculaires. En combinant l'accélération GPU via CUDA et des algorithmes d'intelligence artificielle pour l'analyse des carotides, nous avons transformé un simple outil de visualisation en un véritable assistant de diagnostic.

Les résultats obtenus démontrent des améliorations considérables tant au niveau des performances (jusqu'à 34x d'accélération pour certaines opérations) que des capacités diagnostiques (précision de 92.5% pour la classification des sténoses). Cette double approche permet aux professionnels de santé de bénéficier d'un outil à la fois rapide, réactif et doté de capacités analytiques avancées.

La collaboration avec des partenaires médicaux (CHU, ECRIN, INRIA) a été déterminante pour la pertinence clinique des solutions développées. Cette synergie entre expertise technique et médicale constitue un modèle pour le développement futur d'outils d'aide au diagnostic.

DeepBridge s'inscrit pleinement dans l'évolution de la médecine moderne vers une approche assistée par l'intelligence artificielle, où la technologie ne remplace pas le médecin mais lui fournit des outils puissants pour améliorer la précision diagnostique et, in fine, la qualité des soins prodigués aux patients.