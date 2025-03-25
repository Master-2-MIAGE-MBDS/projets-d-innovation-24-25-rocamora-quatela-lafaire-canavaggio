import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configuration
BASE_DIR = "/Users/moneillon/Programmes/TPI-RQCL-DeepBridge/IACarotideU-Net/imagesdata"
IMAGE_DIR = os.path.join(BASE_DIR, "images")  # Dossier contenant les images pour l'inférence
OUTPUT_DIR = os.path.join(BASE_DIR, "predictions")
MODEL_PATH = os.path.join(BASE_DIR, "carotide_detector_v2.h5")

# Créer les répertoires s'ils n'existent pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paramètres
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1
THRESHOLD = 0.5

# Redéfinir les fonctions personnalisées nécessaires pour charger le modèle
def weighted_binary_crossentropy(y_true, y_pred):
    # Poids pour les pixels positifs (carotides)
    pos_weight = 70  # Utiliser la même valeur que lors de l'entraînement
    
    # Calculer BCE manuellement pour éviter les problèmes de version
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    
    # Appliquer les poids
    weighted_bce = bce * (y_true * pos_weight + (1.0 - y_true))
    
    # Retourner la moyenne
    return tf.reduce_mean(weighted_bce)

# Fonction Dice comme métrique
def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Fonction pour superposer un masque sur une image
def overlay_mask(image, mask, alpha=0.7):
    # Convertir l'image en RGB
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_rgb = cv2.cvtColor((image * 255).astype(np.uint8).reshape(IMG_HEIGHT, IMG_WIDTH), cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = (image * 255).astype(np.uint8)
    
    # Créer une superposition rouge pour les carotides
    overlay = image_rgb.copy()
    binary_mask = (mask > THRESHOLD).astype(np.uint8)
    overlay[binary_mask.reshape(IMG_HEIGHT, IMG_WIDTH) > 0] = [255, 0, 0]  # Rouge
    
    # Mélanger l'image originale et la superposition
    blended = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)
    
    return blended

# Fonction pour prétraiter une image
def preprocess_image(image_path):
    # Lire l'image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de lire l'image: {image_path}")
    
    # Redimensionner
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normaliser
    img = img / 255.0
    
    # Ajouter la dimension du canal
    img = img.reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    
    return img

# Fonction pour prédire et visualiser
def predict_and_visualize(model, image_path, output_path=None):
    # Charger et prétraiter l'image
    img = preprocess_image(image_path)
    
    # Prédire
    pred_mask = model.predict(img)
    
    # Binariser
    binary_pred = (pred_mask > THRESHOLD).astype(np.float32)
    
    # Créer la superposition
    overlay = overlay_mask(img[0], binary_pred[0])
    
    # Afficher
    plt.figure(figsize=(12, 4))
    
    # Image originale
    plt.subplot(1, 3, 1)
    plt.imshow(img[0].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
    plt.title('Image originale')
    plt.axis('off')
    
    # Masque prédit
    plt.subplot(1, 3, 2)
    plt.imshow(binary_pred[0].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='viridis')
    plt.title(f'Masque prédit (seuil: {THRESHOLD})')
    plt.axis('off')
    
    # Superposition
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Superposition')
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualisation sauvegardée: {output_path}")
        
        # Sauvegarder également la superposition séparément
        overlay_path = output_path.replace('.png', '_overlay.png')
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Superposition sauvegardée: {overlay_path}")
    else:
        plt.show()
    
    return binary_pred[0]

# Fonction principale
def main():
    print("Chargement du modèle pré-entraîné...")
    
    # Charger le modèle avec les métriques personnalisées
    custom_objects = {
        'weighted_binary_crossentropy': weighted_binary_crossentropy,
        'dice_coefficient': dice_coefficient
    }
    
    try:
        model = load_model(MODEL_PATH, custom_objects=custom_objects)
        print("Modèle chargé avec succès!")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return
    
    # Obtenir la liste des images à traiter
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
    
    if not image_files:
        print(f"Aucune image PNG trouvée dans {IMAGE_DIR}")
        return
    
    print(f"Traitement de {len(image_files)} images...")
    
    # Traiter chaque image
    for img_file in image_files:
        img_path = os.path.join(IMAGE_DIR, img_file)
        output_path = os.path.join(OUTPUT_DIR, f"pred_{img_file}")
        
        try:
            print(f"Traitement de {img_file}...")
            _ = predict_and_visualize(model, img_path, output_path)
            print(f"Prédiction réussie pour {img_file}")
        except Exception as e:
            print(f"Erreur lors du traitement de {img_file}: {e}")
    
    print("Traitement terminé!")
    print(f"Les résultats ont été sauvegardés dans {OUTPUT_DIR}")

# Pour traiter une seule image spécifique
def process_single_image(model, image_path):
    try:
        print(f"Traitement de l'image: {image_path}")
        filename = os.path.basename(image_path)
        output_path = os.path.join(OUTPUT_DIR, f"pred_{filename}")
        
        binary_pred = predict_and_visualize(model, image_path, output_path)
        return binary_pred
    except Exception as e:
        print(f"Erreur lors du traitement de {image_path}: {e}")
        return None

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Une erreur s'est produite: {e}")
        import traceback
        traceback.print_exc()