import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Configuration
BASE_DIR = "/Users/moneillon/Programmes/TPI-RQCL-DeepBridge/IACarotideU-Net/imagesdata"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "corrected_masks")
OUTPUT_DIR = os.path.join(BASE_DIR, "predictions")
MODEL_PATH = os.path.join(BASE_DIR, "carotide_detector_v2.h5")

# Créer les répertoires s'ils n'existent pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paramètres
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1
BATCH_SIZE = 4
EPOCHS = 50
THRESHOLD = 0.5
CLASS_WEIGHT = 70 

# Fonction de perte personnalisée compatible avec différentes versions de TensorFlow
def weighted_binary_crossentropy(y_true, y_pred):
    # Poids pour les pixels positifs (carotides)
    pos_weight = CLASS_WEIGHT
    
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

# Fonction pour créer le modèle
def create_unet_model():
    # Définir l'entrée
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name='input_layer')
    
    # Encodeur
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Pont
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    
    # Décodeur
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.1)(c6)
    c6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.1)(c7)
    c7 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    # Couche de sortie
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compiler avec la fonction de perte corrigée
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=weighted_binary_crossentropy,
        metrics=['accuracy', dice_coefficient, tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
    )
    
    return model

# Fonction pour charger et prétraiter les données
def load_data():
    print("Chargement des données...")
    
    # Lister tous les fichiers d'images
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
    
    # Trouver les paires image-masque valides
    valid_pairs = []
    
    for img_file in image_files:
        # Extraire le numéro de l'image
        img_num = img_file.replace('carotide', '').replace('.png', '')
        
        # Vérifier différentes possibilités de noms de masque
        possible_mask_names = [
            f"carotide{img_num}_mask_mask.png",  # Double "_mask" possible dans certains fichiers
            f"carotide{img_num}_mask.png"        # Format standard
        ]
        
        for mask_file in possible_mask_names:
            mask_path = os.path.join(MASK_DIR, mask_file)
            if os.path.exists(mask_path):
                # Vérifier si le masque contient des pixels blancs (carotides)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None and np.sum(mask) > 0:  # Le masque contient des pixels blancs
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
    
    for img_file, mask_file in valid_pairs:
        # Charger l'image
        img_path = os.path.join(IMAGE_DIR, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0  # Normalisation
        
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
    
    return X_data, Y_masks, valid_pairs

# Fonction pour augmenter les données
def augment_data(X_train, Y_train, num_augmentations=15):
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
            # 1. Rotation légère (±20°) pour préserver la forme des carotides
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
            
            # 2. Flip horizontal (50% de probabilité)
            if np.random.random() > 0.5:
                rotated_image = cv2.flip(rotated_image, 1)
                rotated_mask = cv2.flip(rotated_mask, 1)
            
            # 3. Ajustements de contraste et luminosité légers
            alpha = 1.0 + np.random.uniform(-0.2, 0.2)  # Contraste
            beta = np.random.uniform(-0.1, 0.1)  # Luminosité
            adjusted_image = np.clip(rotated_image * alpha + beta, 0, 1)
            
            # 4. Zoom léger aléatoire (90-110%)
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
            
            # S'assurer que le masque reste binaire
            rotated_mask = (rotated_mask > 0.5).astype(np.float32)
            
            # 5. Vérifier que les carotides sont encore présentes dans le masque
            if np.sum(rotated_mask) > 10:  # Au moins 10 pixels blancs
                # Ajouter à nos ensembles augmentés
                X_augmented.append(adjusted_image.reshape(IMG_HEIGHT, IMG_WIDTH, 1))
                Y_augmented.append(rotated_mask.reshape(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # Convertir en arrays numpy
    X_augmented = np.array(X_augmented)
    Y_augmented = np.array(Y_augmented)
    
    print(f"Données après augmentation: {X_augmented.shape}")
    return X_augmented, Y_augmented

# Fonction pour visualiser les prédictions
def visualize_results(image, true_mask, pred_mask, title, output_path=None):
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

def save_prediction_overlay(image, pred_mask, filename):
    overlay = overlay_mask(image, pred_mask)
    cv2.imwrite(filename, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# Fonction principale
def main():
    # 1. Charger les données
    try:
        X_data, Y_masks, valid_pairs = load_data()
    except ValueError as e:
        print(f"ERREUR: {e}")
        return
    
    # 2. Diviser en ensembles d'entraînement et de validation
    X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_masks, test_size=0.2, random_state=42)
    
    # 3. Augmenter les données d'entraînement
    X_augmented, Y_augmented = augment_data(X_train, Y_train, num_augmentations=15)
    
    # 4. Créer et entraîner le modèle
    model = create_unet_model()
    print(model.summary())
    
    # Callbacks pour l'entraînement
    callbacks = [
        ModelCheckpoint(MODEL_PATH, verbose=1, save_best_only=True, monitor='val_dice_coefficient', mode='max'),
        EarlyStopping(patience=20, verbose=1, monitor='val_dice_coefficient', mode='max', restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, verbose=1, monitor='val_dice_coefficient', mode='max')
    ]
    
    # Entraîner le modèle
    print("Début de l'entraînement...")
    print(f"Poids de classe utilisé pour les carotides: {CLASS_WEIGHT}")
    
    history = model.fit(
        X_augmented, Y_augmented,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # 5. Visualiser l'historique d'entraînement
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
    plt.savefig(os.path.join(BASE_DIR, 'training_history_v2.png'))
    
    # 6. Évaluer le modèle
    print("Évaluation du modèle...")
    evaluation = model.evaluate(X_val, Y_val)
    metrics = model.metrics_names
    for i, metric in enumerate(metrics):
        print(f"{metric}: {evaluation[i]}")
    
    # 7. Prédire sur l'ensemble de validation
    print("Génération des prédictions...")
    Y_pred_val = model.predict(X_val)
    
    # 8. Visualiser quelques résultats
    os.makedirs(os.path.join(BASE_DIR, 'validation_results_v2'), exist_ok=True)
    
    num_samples = min(5, len(X_val))
    for i in range(num_samples):
        # Prédire le masque
        binary_pred = (Y_pred_val[i] > THRESHOLD).astype(np.uint8)
        
        # Visualiser
        visualize_results(
            X_val[i],
            Y_val[i],
            binary_pred,
            f"Seuil: {THRESHOLD}",
            os.path.join(BASE_DIR, f'validation_results_v2/sample_{i}.png')
        )
        
        # Sauvegarder avec superposition
        save_prediction_overlay(
            X_val[i],
            binary_pred,
            os.path.join(BASE_DIR, f'validation_results_v2/overlay_{i}.png')
        )
    
    print("Traitement terminé!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Une erreur s'est produite: {e}")
        import traceback
        traceback.print_exc()
