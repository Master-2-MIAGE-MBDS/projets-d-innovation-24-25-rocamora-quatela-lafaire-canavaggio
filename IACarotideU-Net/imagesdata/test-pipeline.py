import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import random
from pathlib import Path

# 1. CONFIGURATION
# Définissez vos chemins d'accès
INPUT_DIR = "/Users/moneillon/Programmes/TPI-RQCL-DeepBridge/IACarotideU-Net/imagesdata/images"        
MASK_DIR = "/Users/moneillon/Programmes/TPI-RQCL-DeepBridge/IACarotideU-Net/imagesdata/masks"          # Dossier contenant les masques annotés
OUTPUT_DIR = "./imagesbaba"    # Dossier où sauvegarder les images augmentées

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)

# Nombre d'augmentations par image
NUM_AUGMENTATIONS = 5  # Chaque image originale générera 5 nouvelles images

# 2. FONCTIONS D'AUGMENTATION
def create_augmentation_pipeline():
    """Création du pipeline d'augmentation avec Albumentations"""
    return A.Compose([
        # Rotation avec probabilité de 100%
        A.Rotate(limit=30, p=1.0),
        
        # Autres augmentations utiles (facultatif)
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])

def apply_augmentation(image, mask, transform):
    """Appliquer une transformation à une image et son masque"""
    augmented = transform(image=image, mask=mask)
    return augmented['image'], augmented['mask']

# 3. CHARGEMENT ET AUGMENTATION DES IMAGES
def augment_dataset():
    """Charger et augmenter toutes les images du dataset"""
    # Récupérer la liste des fichiers d'images
    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.tif'))]
    print(f"Nombre d'images trouvées: {len(image_files)}")
    
    # Créer le pipeline d'augmentation
    transform = create_augmentation_pipeline()
    
    # Pour chaque paire image/masque
    for img_file in image_files:
        # Construire les chemins complets
        img_path = os.path.join(INPUT_DIR, img_file)
        mask_file = img_file  # Ajustez si vos masques ont un nom différent
        mask_path = os.path.join(MASK_DIR, mask_file)
        
        # Vérifier que le masque existe
        if not os.path.exists(mask_path):
            print(f"Masque non trouvé pour {img_file}, on passe...")
            continue
        
        # Charger l'image et le masque
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Impossible de charger l'image {img_file}")
            continue
            
        # Convertir l'image BGR en RGB si nécessaire
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Impossible de charger le masque {mask_file}")
            continue
        
        # Sauvegarder l'image et le masque originaux
        original_img_out = os.path.join(OUTPUT_DIR, "images", f"original_{img_file}")
        original_mask_out = os.path.join(OUTPUT_DIR, "masks", f"original_{mask_file}")
        cv2.imwrite(original_img_out, image if len(image.shape) < 3 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(original_mask_out, mask)
        
        # Générer plusieurs versions augmentées
        for i in range(NUM_AUGMENTATIONS):
            # Appliquer l'augmentation
            aug_img, aug_mask = apply_augmentation(image, mask, transform)
            
            # Définir les noms de fichiers de sortie
            aug_img_filename = f"{Path(img_file).stem}_aug{i}{Path(img_file).suffix}"
            aug_mask_filename = f"{Path(mask_file).stem}_aug{i}{Path(mask_file).suffix}"
            
            # Chemins complets
            aug_img_path = os.path.join(OUTPUT_DIR, "images", aug_img_filename)
            aug_mask_path = os.path.join(OUTPUT_DIR, "masks", aug_mask_filename)
            
            # Sauvegarder les versions augmentées
            cv2.imwrite(aug_img_path, aug_img if len(aug_img.shape) < 3 else cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(aug_mask_path, aug_mask)
            
            print(f"Augmentation {i+1}/{NUM_AUGMENTATIONS} créée pour {img_file}")
        
        print(f"Terminé pour {img_file}")

# 4. VISUALISATION DES RÉSULTATS (OPTIONNEL)
def visualize_results(num_samples=3):
    """Visualiser quelques exemples d'images augmentées"""
    # Récupérer quelques paires image/masque augmentées aléatoirement
    aug_images = [f for f in os.listdir(os.path.join(OUTPUT_DIR, "images")) if "aug" in f]
    
    if not aug_images:
        print("Aucune image augmentée trouvée.")
        return
    
    # Sélectionner quelques exemples aléatoires
    samples = random.sample(aug_images, min(num_samples, len(aug_images)))
    
    plt.figure(figsize=(15, 5 * len(samples)))
    
    for i, sample in enumerate(samples):
        # Charger l'image augmentée
        img_path = os.path.join(OUTPUT_DIR, "images", sample)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Charger le masque correspondant
        mask_name = sample.replace("images", "masks")
        mask_path = os.path.join(OUTPUT_DIR, "masks", mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # Afficher
        plt.subplot(len(samples), 2, i*2+1)
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        plt.title(f"Image augmentée: {sample}")
        plt.axis('off')
        
        plt.subplot(len(samples), 2, i*2+2)
        plt.imshow(mask, cmap='viridis')
        plt.title(f"Masque augmenté: {mask_name}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 5. EXÉCUTION PRINCIPALE
if __name__ == "__main__":
    print("Démarrage de l'augmentation du dataset...")
    augment_dataset()
    print(f"Augmentation terminée! Les résultats sont dans: {OUTPUT_DIR}")
    
    # Visualiser quelques exemples (décommentez pour utiliser)
    # visualize_results(3)


# EXEMPLE D'UTILISATION RAPIDE AVEC UN SEUL FICHIER:
"""
# Pour tester rapidement avec une seule image:
image_path = "chemin/vers/votre/image.png"
mask_path = "chemin/vers/votre/masque.png"

# Charger l'image et le masque
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

# Créer une transformation avec rotation
transform = A.Compose([
    A.Rotate(limit=30, p=1.0)  # Rotation entre -30 et +30 degrés
])

# Appliquer la transformation
augmented = transform(image=image, mask=mask)
rotated_image = augmented['image']
rotated_mask = augmented['mask']

# Afficher les résultats
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(rotated_image, cmap='gray')
plt.title('Image pivotée')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(rotated_mask, cmap='viridis')
plt.title('Masque pivoté')
plt.axis('off')

plt.tight_layout()
plt.show()
"""
