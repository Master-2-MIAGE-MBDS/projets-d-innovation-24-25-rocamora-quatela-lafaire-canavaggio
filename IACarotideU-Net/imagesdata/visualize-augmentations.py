import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, KFold
import glob
import time
import datetime

# Configuration
BASE_DIR = "/Users/moneillon/Programmes/TPI-RQCL-DeepBridge/IACarotideU-Net/imagesdata"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "corrected_masks")
OUTPUT_DIR = os.path.join(BASE_DIR, "predictions")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEST_DIR = os.path.join(BASE_DIR, "test_images")  # Dossier pour les images de test
RESULTS_DIR = os.path.join(BASE_DIR, "results")   # Dossier pour les résultats

# Créer les répertoires nécessaires
for directory in [OUTPUT_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Paramètres
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1
BATCH_SIZE = 4
EPOCHS = 30
THRESHOLD = 0.5
CLASS_WEIGHT = 80.0  # Valeur initiale, sera ajustée automatiquement
USE_CROSS_VALIDATION = True  # Activer la validation croisée
N_FOLDS = 5  # Nombre de folds pour la validation croisée
RANDOM_SEED = 42  # Pour la reproductibilité

# Timestamp pour les noms de fichiers uniques
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ==============================================================================
# FONCTIONS DE PERTE ET MÉTRIQUES
# ==============================================================================

def dice_loss(y_true, y_pred, smooth=1.0):
    """Fonction de perte Dice (1 - coefficient Dice)"""
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return 1 - dice

def weighted_bce(y_true, y_pred, pos_weight=CLASS_WEIGHT):
    """BCE pondérée pour gérer le déséquilibre des classes"""
    # Stabiliser les prédictions
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculer BCE manuellement
    bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    
    # Appliquer les poids
    weighted_bce = bce * (y_true * pos_weight + (1.0 - y_true))
    
    return tf.reduce_mean(weighted_bce)

def combined_loss(y_true, y_pred, bce_weight=0.7, dice_weight=0.3, pos_weight=CLASS_WEIGHT):
    """Combinaison de BCE pondérée et Dice Loss"""
    bce = weighted_bce(y_true, y_pred, pos_weight)
    dice = dice_loss(y_true, y_pred)
    return bce_weight * bce + dice_weight * dice

def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Coefficient Dice pour mesurer la performance"""
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def specificity(y_true, y_pred):
    """Calcule la spécificité (true negative rate)"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    y_pred_binary = tf.cast(y_pred_f > 0.5, tf.float32)
    true_negatives = tf.keras.backend.sum((1 - y_true_f) * (1 - y_pred_binary))
    actual_negatives = tf.keras.backend.sum(1 - y_true_f)
    return true_negatives / (actual_negatives + tf.keras.backend.epsilon())

def sensitivity(y_true, y_pred):
    """Calcule la sensibilité (true positive rate / recall)"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    y_pred_binary = tf.cast(y_pred_f > 0.5, tf.float32)
    true_positives = tf.keras.backend.sum(y_true_f * y_pred_binary)
    actual_positives = tf.keras.backend.sum(y_true_f)
    return true_positives / (actual_positives + tf.keras.backend.epsilon())

# ==============================================================================
# PRÉTRAITEMENT D'IMAGES
# ==============================================================================

def preprocess_image(image):
    """Amélioration du prétraitement des images"""
    # Normalisation
    if image.max() > 1.0:
        image = image / 255.0
    
    # Amélioration du contraste par CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Préparer l'image pour CLAHE
        img_to_process = (image * 255).astype(np.uint8)
        if len(img_to_process.shape) == 3:
            img_to_process = img_to_process.reshape(img_to_process.shape[0], img_to_process.shape[1])
        
        # Appliquer CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_to_process)
        
        # Normaliser à nouveau
        enhanced = enhanced / 255.0
        
        # Remettre en forme si nécessaire
        if len(image.shape) == 3:
            enhanced = enhanced.reshape(image.shape)
            
        return enhanced
    
    return image

def load_and_preprocess_data():
    """Chargement et prétraitement améliorés des données"""
    print("Chargement et prétraitement des données...")
    
    # Lister tous les fichiers d'images
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')])
    
    # Trouver les paires image-masque valides
    valid_pairs = []
    
    for img_file in image_files:
        # Extraire le numéro de l'image
        img_num = img_file.replace('carotide', '').replace('.png', '')
        
        # Vérifier différentes possibilités de noms de masque
        possible_mask_names = [
            f"carotide{img_num}_mask_mask.png",  # Double "_mask" possible 
            f"carotide{img_num}_mask.png"        # Format standard
        ]
        
        for mask_file in possible_mask_names:
            mask_path = os.path.join(MASK_DIR, mask_file)
            if os.path.exists(mask_path):
                # Vérifier si le masque contient des pixels blancs (carotides)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None and np.sum(mask) > 0:
                    valid_pairs.append((img_file, mask_file))
                    print(f"Paire valide trouvée: {img_file} - {mask_file}")
                    break
                else:
                    print(f"ATTENTION: Masque vide ignoré: {mask_file}")
    
    print(f"Nombre de paires image-masque valides: {len(valid_pairs)}")
    
    if len(valid_pairs) == 0:
        raise ValueError("Aucune paire image-masque valide trouvée. Vérifiez vos masques!")
    
    # Charger les images et masques valides
    X_data = []
    Y_masks = []
    filenames = []  # Pour garder une trace des noms de fichiers
    
    for img_file, mask_file in valid_pairs:
        # Charger l'image
        img_path = os.path.join(IMAGE_DIR, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        # Prétraitement amélioré
        img = preprocess_image(img)
        
        # Charger le masque
        mask_path = os.path.join(MASK_DIR, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask = (mask > 127).astype(np.float32)  # Binarisation stricte
        
        # Afficher des statistiques sur le masque
        white_pixels = np.sum(mask)
        total_pixels = IMG_HEIGHT * IMG_WIDTH
        percentage = (white_pixels / total_pixels) * 100
        print(f"Masque {mask_file}: {white_pixels} pixels blancs ({percentage:.4f}% de l'image)")
        
        X_data.append(img)
        Y_masks.append(mask)
        filenames.append(img_file)
    
    # Calculer le poids de classe recommandé basé sur le déséquilibre des classes
    if Y_masks:
        total_white_pixels = sum(np.sum(mask) for mask in Y_masks)
        total_pixels = len(Y_masks) * IMG_HEIGHT * IMG_WIDTH
        ratio = (total_pixels - total_white_pixels) / total_white_pixels if total_white_pixels > 0 else 100
        print(f"Ratio de déséquilibre des classes (noir/blanc): {ratio:.2f}")
        print(f"Poids de classe recommandé: {ratio:.2f}")
        global CLASS_WEIGHT
        CLASS_WEIGHT = min(max(ratio, 10), 100)  # Limiter entre 10 et 100
    
    # Convertir en arrays numpy
    X_data = np.array(X_data).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    Y_masks = np.array(Y_masks).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    
    print(f"Forme des données: {X_data.shape}")
    print(f"Forme des masques: {Y_masks.shape}")
    
    return X_data, Y_masks, filenames, valid_pairs

# ==============================================================================
# AUGMENTATION DE DONNÉES
# ==============================================================================

def augment_data(X_train, Y_train, num_augmentations=15):
    """Augmentation de données améliorée"""
    print("Augmentation des données...")
    
    X_augmented = []
    Y_augmented = []
    
    # Ajouter les images originales
    X_augmented.extend(X_train)
    Y_augmented.extend(Y_train)
    
    # Pour chaque image et masque
    for i in range(len(X_train)):
        image = X_train[i].reshape(IMG_HEIGHT, IMG_WIDTH)
        mask = Y_train[i].reshape(IMG_HEIGHT, IMG_WIDTH)
        
        # Vérifier que le masque contient des carotides
        if np.sum(mask) < 10:
            print(f"ATTENTION: Masque #{i} presque vide, ignoré pour l'augmentation")
            continue
        
        # Créer plusieurs versions augmentées
        for j in range(num_augmentations):
            # Combinaison aléatoire d'augmentations
            # Rotation aléatoire (±20°)
            angle = np.random.uniform(-20, 20)
            M = cv2.getRotationMatrix2D((IMG_WIDTH//2, IMG_HEIGHT//2), angle, 1)
            
            rotated_image = cv2.warpAffine(
                image, 
                M, 
                (IMG_WIDTH, IMG_HEIGHT),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            rotated_mask = cv2.warpAffine(
                mask, 
                M, 
                (IMG_WIDTH, IMG_HEIGHT),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT
            )
            
            # Flip horizontal (50% de probabilité)
            if np.random.random() > 0.5:
                rotated_image = cv2.flip(rotated_image, 1)
                rotated_mask = cv2.flip(rotated_mask, 1)
            
            # Ajustements de contraste et luminosité légers
            alpha = 1.0 + np.random.uniform(-0.2, 0.2)  # Contraste
            beta = np.random.uniform(-0.1, 0.1)  # Luminosité
            adjusted_image = np.clip(rotated_image * alpha + beta, 0, 1)
            
            # Zoom léger aléatoire (90-110%)
            if np.random.random() > 0.5:
                zoom_factor = np.random.uniform(0.9, 1.1)
                h, w = rotated_image.shape
                h_new, w_new = int(zoom_factor * h), int(zoom_factor * w)
                
                # Calculer les coordonnées pour le recadrage
                h_start = max(0, (h_new - h) // 2)
                w_start = max(0, (w_new - w) // 2)
                h_end = min(h_new, h_start + h)
                w_end = min(w_new, w_start + w)
                
                if zoom_factor > 1:  # Zoom out
                    adjusted_image = cv2.resize(adjusted_image, (w_new, h_new))
                    adjusted_image = adjusted_image[h_start:h_end, w_start:w_end]
                    adjusted_image = cv2.resize(adjusted_image, (w, h))
                    
                    rotated_mask = cv2.resize(rotated_mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
                    rotated_mask = rotated_mask[h_start:h_end, w_start:w_end]
                    rotated_mask = cv2.resize(rotated_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                else:  # Zoom in
                    # Calculer les coordonnées pour le recadrage
                    h_start = max(0, (h - h_new) // 2)
                    w_start = max(0, (w - w_new) // 2)
                    h_end = min(h, h_start + h_new)
                    w_end = min(w, w_start + w_new)
                    
                    adjusted_image = adjusted_image[h_start:h_end, w_start:w_end]
                    adjusted_image = cv2.resize(adjusted_image, (w, h))
                    
                    rotated_mask = rotated_mask[h_start:h_end, w_start:w_end]
                    rotated_mask = cv2.resize(rotated_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Ajout de bruit gaussien (25% de probabilité)
            if np.random.random() > 0.75:
                noise = np.random.normal(0, 0.03, adjusted_image.shape)
                adjusted_image = np.clip(adjusted_image + noise, 0, 1)
            
            # S'assurer que le masque reste binaire
            rotated_mask = (rotated_mask > 0.5).astype(np.float32)
            
            # Vérifier que les carotides sont encore présentes dans le masque
            if np.sum(rotated_mask) > 10:  # Au moins 10 pixels blancs
                # Ajouter à nos ensembles augmentés
                X_augmented.append(adjusted_image.reshape(IMG_HEIGHT, IMG_WIDTH, 1))
                Y_augmented.append(rotated_mask.reshape(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # Convertir en arrays numpy
    X_augmented = np.array(X_augmented)
    Y_augmented = np.array(Y_augmented)
    
    print(f"Données après augmentation: {X_augmented.shape}")
    return X_augmented, Y_augmented

# ==============================================================================
# ARCHITECTURE DU MODÈLE
# ==============================================================================

def create_improved_unet():
    """Crée un U-Net amélioré avec BatchNormalization et plus de filtres"""
    # Définir l'entrée
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name='input_layer')
    
    # Bloc d'encodeur 1
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    # Bloc d'encodeur 2
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bloc d'encodeur 3
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Bloc d'encodeur 4
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Pont
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    
    # Bloc de décodeur 1
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)
    
    # Bloc de décodeur 2
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)
    
    # Bloc de décodeur 3
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = BatchNormalization()(c8)
    
    # Bloc de décodeur 4
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    c9 = BatchNormalization()(c9)
    
    # Couche de sortie
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    # Créer le modèle
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compiler avec la perte combinée
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, bce_weight=0.7, dice_weight=0.3, pos_weight=CLASS_WEIGHT),
        metrics=['accuracy', dice_coefficient, sensitivity, specificity, 
                 tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1], name='iou')]
    )
    
    return model

# ==============================================================================
# VISUALISATION ET ÉVALUATION
# ==============================================================================

def visualize_results(image, true_mask, pred_mask, title, output_path=None):
    """Visualise une image, son masque réel et sa prédiction"""
    plt.figure(figsize=(12, 4))
    
    # Image originale
    plt.subplot(1, 3, 1)
    plt.imshow(image.reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
    plt.title('Image originale')
    plt.axis('off')
    
    # Masque réel
    plt.subplot(1, 3, 2)
    plt.imshow(true_mask.reshape(IMG_HEIGHT, IMG_WIDTH), cmap='viridis')
    plt.title('Masque réel')
    plt.axis('off')
    
    # Masque prédit
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask.reshape(IMG_HEIGHT, IMG_WIDTH), cmap='viridis')
    plt.title(f'Masque prédit ({title})')
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualisation sauvegardée: {output_path}")
    else:
        plt.show()

def overlay_mask(image, mask, alpha=0.7):
    """Superpose un masque sur une image"""
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

def save_prediction_overlay(image, pred_mask, filename):
    """Sauvegarde une image avec la prédiction superposée"""
    overlay = overlay_mask(image, pred_mask)
    cv2.imwrite(filename, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def evaluate_model(model, X_test, Y_test, test_filenames, output_dir):
    """Évalue le modèle sur les données de test"""
    print("Évaluation du modèle...")
    
    # Créer un dossier pour les résultats d'évaluation
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Évaluer le modèle
    evaluation = model.evaluate(X_test, Y_test, verbose=1)
    metrics = model.metrics_names
    
    # Afficher et enregistrer les métriques
    with open(os.path.join(eval_dir, "metrics.txt"), "w") as f:
        for i, metric in enumerate(metrics):
            print(f"{metric}: {evaluation[i]:.4f}")
            f.write(f"{metric}: {evaluation[i]:.4f}\n")
    
    # Prédire les masques
    Y_pred = model.predict(X_test)
    
    # Visualiser les résultats
    for i in range(len(X_test)):
        binary_pred = (Y_pred[i] > THRESHOLD).astype(np.uint8)
        
        # Calculer le Dice pour cette image
        dice = dice_coefficient(Y_test[i], Y_pred[i]).numpy()
        
        # Visualiser
        output_path = os.path.join(eval_dir, f"{test_filenames[i].replace('.png', '')}_result.png")
        visualize_results(
            X_test[i],
            Y_test[i],
            binary_pred,
            f"Dice: {dice:.4f}",
            output_path
        )
        
        # Sauvegarder avec superposition
        overlay_path = os.path.join(eval_dir, f"{test_filenames[i].replace('.png', '')}_overlay.png")
        save_prediction_overlay(
            X_test[i],
            binary_pred,
            overlay_path
        )
    
    # Créer un tableau de résultats détaillé par image
    detailed_metrics = []
    for i in range(len(X_test)):
        pred = Y_pred[i]
        true = Y_test[i]
        
        # Calculer les métriques pour cette image
        dice = dice_coefficient(true, pred).numpy()
        sens = sensitivity(true, pred).numpy()
        spec = specificity(true, pred).numpy()
        
        detailed_metrics.append({
            'filename': test_filenames[i],
            'dice': dice,
            'sensitivity': sens,
            'specificity': spec,
            'white_pixels_true': np.sum(true > 0.5),
            'white_pixels_pred': np.sum(pred > 0.5)
        })
    
    # Sauvegarder les métriques détaillées
    with open(os.path.join(eval_dir, "detailed_metrics.txt"), "w") as f:
        f.write("Filename,Dice,Sensitivity,Specificity,WhitePixelsTrue,WhitePixelsPred\n")
        for m in detailed_metrics:
            f.write(f"{m['filename']},{m['dice']:.4f},{m['sensitivity']:.4f},{m['specificity']:.4f},{m['white_pixels_true']},{m['white_pixels_pred']}\n")
    
    return evaluation, detailed_metrics

def predict_on_new_images(model, test_dir, output_dir):
    """Prédiction sur de nouvelles images"""
    print(f"Prédiction sur des nouvelles images dans {test_dir}...")
    
    # Créer un dossier pour les résultats
    predictions_dir = os.path.join(output_dir, "new_predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Lister toutes les images de test
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not test_images:
        print(f"Aucune image trouvée dans {test_dir}")
        return
    
    for img_file in test_images:
        img_path = os.path.join(test_dir, img_file)
        
        # Charger et prétraiter l'image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Impossible de charger l'image: {img_path}")
            continue
        
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = preprocess_image(image)
        
        # Prédire le masque
        pred = model.predict(image.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1))[0]
        binary_pred = (pred > THRESHOLD).astype(np.uint8)
        
        # Sauvegarder le masque prédit
        mask_filename = os.path.join(predictions_dir, f"{os.path.splitext(img_file)[0]}_mask.png")
        cv2.imwrite(mask_filename, binary_pred.reshape(IMG_HEIGHT, IMG_WIDTH) * 255)
        
        # Sauvegarder l'overlay
        overlay_filename = os.path.join(predictions_dir, f"{os.path.splitext(img_file)[0]}_overlay.png")
        save_prediction_overlay(image, binary_pred, overlay_filename)
        
        print(f"Prédiction sauvegardée pour {img_file}")
    
    print(f"Toutes les prédictions ont été sauvegardées dans {predictions_dir}")

# ==============================================================================
# FONCTIONS PRINCIPALES
# ==============================================================================

def train_single_fold(X_train, Y_train, X_val, Y_val, fold_num=None, suffix=""):
    """Entraîne un modèle sur un fold"""
    # Augmenter les données d'entraînement
    X_augmented, Y_augmented = augment_data(X_train, Y_train, num_augmentations=15)
    
    # Créer le modèle
    model = create_improved_unet()
    
    # Définir le chemin pour sauvegarder le modèle
    if fold_num is not None:
        model_path = os.path.join(MODELS_DIR, f"carotid_model_fold{fold_num}_{suffix}.h5")
    else:
        model_path = os.path.join(MODELS_DIR, f"carotid_model_{suffix}.h5")
    
    # Callbacks pour l'entraînement
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_dice_coefficient', mode='max'),
        EarlyStopping(patience=20, verbose=1, monitor='val_dice_coefficient', mode='max', restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, verbose=1, monitor='val_dice_coefficient', mode='max')
    ]
    
    # Entraîner le modèle
    fold_text = f"Fold {fold_num} " if fold_num is not None else ""
    print(f"Début de l'entraînement {fold_text}...")
    print(f"Poids de classe utilisé pour les carotides: {CLASS_WEIGHT}")
    
    history = model.fit(
        X_augmented, Y_augmented,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Charger le meilleur modèle
    model = load_model(model_path, custom_objects={
        'combined_loss': combined_loss,
        'dice_coefficient': dice_coefficient,
        'sensitivity': sensitivity,
        'specificity': specificity
    })
    
    return model, history

def train_with_cross_validation(X_data, Y_masks, filenames):
    """Entraîne le modèle avec validation croisée k-fold"""
    print(f"Démarrage de l'entraînement avec validation croisée ({N_FOLDS} folds)...")
    
    # Définir la validation croisée
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Pour stocker les métriques de chaque fold
    fold_metrics = []
    histories = []
    
    # Dossier pour les résultats de validation croisée
    cv_dir = os.path.join(RESULTS_DIR, "cross_validation")
    os.makedirs(cv_dir, exist_ok=True)
    
    # Pour chaque fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_data)):
        print(f"\n===== Fold {fold+1}/{N_FOLDS} =====")
        
        # Séparer les données d'entraînement et de validation
        X_train, X_val = X_data[train_idx], X_data[val_idx]
        Y_train, Y_val = Y_masks[train_idx], Y_masks[val_idx]
        train_filenames = [filenames[i] for i in train_idx]
        val_filenames = [filenames[i] for i in val_idx]
        
        print(f"Entraînement: {len(X_train)} images, Validation: {len(X_val)} images")
        
        # Entraîner le modèle sur ce fold
        fold_model, history = train_single_fold(X_train, Y_train, X_val, Y_val, fold_num=fold+1, suffix=TIMESTAMP)
        
        # Évaluer le modèle sur les données de validation
        fold_eval_dir = os.path.join(cv_dir, f"fold_{fold+1}")
        os.makedirs(fold_eval_dir, exist_ok=True)
        
        evaluation, detailed_metrics = evaluate_model(fold_model, X_val, Y_val, val_filenames, fold_eval_dir)
        
        # Stocker les métriques et l'historique
        fold_metrics.append({
            'fold': fold+1,
            'evaluation': evaluation,
            'metrics_names': fold_model.metrics_names,
            'detailed_metrics': detailed_metrics
        })
        histories.append(history)
        
        # Visualiser l'historique d'entraînement pour ce fold
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title(f'Fold {fold+1} - Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title(f'Fold {fold+1} - Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(history.history['dice_coefficient'], label='Train')
        plt.plot(history.history['val_dice_coefficient'], label='Validation')
        plt.title(f'Fold {fold+1} - Dice Coefficient')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(fold_eval_dir, "training_history.png"))
    
    # Calculer et sauvegarder les métriques moyennes sur tous les folds
    mean_metrics = {}
    for metric in fold_metrics[0]['metrics_names']:
        values = [fold['evaluation'][fold['metrics_names'].index(metric)] for fold in fold_metrics]
        mean_metrics[metric] = np.mean(values)
    
    with open(os.path.join(cv_dir, "mean_metrics.txt"), "w") as f:
        f.write("Cross-Validation Mean Metrics:\n")
        for metric, value in mean_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # Trouver le meilleur fold
    best_fold_idx = np.argmax([fold['evaluation'][fold['metrics_names'].index('dice_coefficient')] for fold in fold_metrics])
    best_fold = fold_metrics[best_fold_idx]
    
    with open(os.path.join(cv_dir, "best_fold.txt"), "w") as f:
        f.write(f"Best Fold: {best_fold['fold']}\n")
        for i, metric in enumerate(best_fold['metrics_names']):
            f.write(f"{metric}: {best_fold['evaluation'][i]:.4f}\n")
    
    # Retourner le modèle du meilleur fold
    best_model = load_model(os.path.join(MODELS_DIR, f"carotid_model_fold{best_fold['fold']}_{TIMESTAMP}.h5"), 
                          custom_objects={
                              'combined_loss': combined_loss,
                              'dice_coefficient': dice_coefficient,
                              'sensitivity': sensitivity,
                              'specificity': specificity
                          })
    
    return best_model, fold_metrics, histories

def train_baseline_model(X_data, Y_masks, filenames):
    """Entraîne un modèle de base sans validation croisée"""
    print("Démarrage de l'entraînement du modèle de base...")
    
    # Diviser les données en ensembles d'entraînement et de validation
    X_train, X_val, Y_train, Y_val, train_filenames, val_filenames = train_test_split(
        X_data, Y_masks, filenames, test_size=0.2, random_state=RANDOM_SEED
    )
    
    print(f"Entraînement: {len(X_train)} images, Validation: {len(X_val)} images")
    
    # Entraîner le modèle
    model, history = train_single_fold(X_train, Y_train, X_val, Y_val, suffix=TIMESTAMP)
    
    # Évaluer le modèle
    eval_dir = os.path.join(RESULTS_DIR, "baseline_evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    evaluation, detailed_metrics = evaluate_model(model, X_val, Y_val, val_filenames, eval_dir)
    
    # Visualiser l'historique d'entraînement
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['dice_coefficient'], label='Train')
    plt.plot(history.history['val_dice_coefficient'], label='Validation')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "training_history.png"))
    
    return model, history, evaluation, X_val, Y_val, val_filenames

def main():
    """Fonction principale"""
    
    start_time = time.time()
    
    # 1. Créer les répertoires nécessaires
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # 2. Charger et prétraiter les données
    X_data, Y_masks, filenames, valid_pairs = load_and_preprocess_data()
    
    # 3. Entraîner le modèle
    if USE_CROSS_VALIDATION:
        # Entraîner avec validation croisée
        best_model, fold_metrics, histories = train_with_cross_validation(X_data, Y_masks, filenames)
        
        # Sauvegarder le meilleur modèle final
        best_model.save(os.path.join(MODELS_DIR, f"best_carotid_model_{TIMESTAMP}.h5"))
        print(f"Meilleur modèle sauvegardé: best_carotid_model_{TIMESTAMP}.h5")
        
        # Utiliser ce modèle pour les prédictions
        model = best_model
    else:
        # Entraîner un modèle de base
        model, history, evaluation, X_val, Y_val, val_filenames = train_baseline_model(X_data, Y_masks, filenames)
    
    # 4. Prédire sur des nouvelles images
    if os.path.exists(TEST_DIR) and len(os.listdir(TEST_DIR)) > 0:
        predict_on_new_images(model, TEST_DIR, RESULTS_DIR)
    
    elapsed_time = time.time() - start_time
    print(f"Temps d'exécution total: {elapsed_time:.2f} secondes ({elapsed_time/60:.2f} minutes)")
    print("Traitement terminé!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Une erreur s'est produite: {e}")
        import traceback
        traceback.print_exc()