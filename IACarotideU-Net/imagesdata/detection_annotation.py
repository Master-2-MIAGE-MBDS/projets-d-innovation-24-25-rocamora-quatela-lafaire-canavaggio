import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def detect_annotation_colors(image_path, base_color_hex="#aa0003", k=5):
    """
    Analyse une image pour trouver les couleurs qui pourraient être des annotations,
    en se basant sur une couleur de référence.
    
    Args:
        image_path: Chemin vers l'image
        base_color_hex: Couleur de référence (hexadécimal)
        k: Nombre de clusters à rechercher
    
    Returns:
        Liste des couleurs potentielles d'annotation (BGR)
    """
    # Charger l'image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Impossible de charger l'image: {image_path}")
        return []
    
    # Convertir le code hexadécimal en BGR
    base_r = int(base_color_hex[1:3], 16)
    base_g = int(base_color_hex[3:5], 16)
    base_b = int(base_color_hex[5:7], 16)
    base_color = np.array([base_b, base_g, base_r])
    
    # Aplatir l'image pour avoir une liste de pixels
    pixels = image.reshape(-1, 3)
    
    # Filtrer les pixels gris
    non_gray_pixels = []
    for pixel in pixels:
        b, g, r = pixel
        # Si ce n'est pas un pixel gris (R ≈ G ≈ B)
        if not (abs(r - g) < 15 and abs(r - b) < 15 and abs(g - b) < 15):
            # Si ce n'est pas trop foncé ou trop clair
            if 10 < r < 245 or 10 < g < 245 or 10 < b < 245:
                non_gray_pixels.append([b, g, r])
    
    # S'il n'y a pas assez de pixels colorés, retourner la couleur de base
    if len(non_gray_pixels) < 100:
        print(f"Pas assez de pixels colorés dans l'image, utilisation de la couleur de base")
        return [base_color]
    
    # Convertir en array numpy
    non_gray_pixels = np.array(non_gray_pixels)
    
    # Appliquer K-means pour trouver les couleurs dominantes
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(non_gray_pixels)
    
    # Obtenir les centres des clusters (couleurs dominantes)
    colors = kmeans.cluster_centers_.astype(int)
    
    # Trier les couleurs par distance à la couleur de base (les plus proches en premier)
    distances = [np.sqrt(np.sum((color - base_color)**2)) for color in colors]
    sorted_indices = np.argsort(distances)
    sorted_colors = colors[sorted_indices]
    
    # Afficher les couleurs trouvées
    print("Couleurs dominantes (BGR) dans l'image, triées par proximité avec #aa0003:")
    for i, color in enumerate(sorted_colors):
        b, g, r = color
        hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        print(f"Couleur {i+1}: BGR({b}, {g}, {r}) / RGB({r}, {g}, {b}) / HEX: {hex_color} - Distance: {distances[sorted_indices[i]]:.1f}")
    
    # Sélectionner les couleurs qui semblent être des annotations rouges
    # (Proximité avec la couleur de base ou dominance de rouge)
    annotation_colors = []
    for color in sorted_colors:
        b, g, r = color
        # Si la couleur est proche de la couleur de base ou a une forte dominance rouge
        distance = np.sqrt(np.sum((color - base_color)**2))
        if distance < 100 or (r > b + 50 and r > g + 50):
            annotation_colors.append(color)
    
    # Si aucune couleur d'annotation n'est trouvée, utiliser la couleur la plus proche de la base
    if not annotation_colors:
        annotation_colors.append(sorted_colors[0])
    
    return annotation_colors

def create_mask_from_colors(image_path, output_path, colors, tolerance=30):
    """
    Crée un masque binaire basé sur plusieurs couleurs avec tolérance.
    
    Args:
        image_path: Chemin vers l'image
        output_path: Chemin pour sauvegarder le masque
        colors: Liste de couleurs BGR à détecter
        tolerance: Tolérance pour la détection de couleur
    
    Returns:
        bool: True si le masque a été créé avec succès
    """
    # Charger l'image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Impossible de charger l'image: {image_path}")
        return False
    
    # Créer un masque vide
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Pour chaque couleur, détecter les pixels correspondants
    for color in colors:
        color_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Deux approches: distance euclidienne et HSV
        # 1. Distance euclidienne
        for y in range(height):
            for x in range(width):
                pixel = image[y, x]
                # Calculer la distance entre le pixel et la couleur cible
                distance = np.sqrt(np.sum((pixel - color)**2))
                if distance < tolerance:
                    color_mask[y, x] = 255
        
        # 2. Approche HSV (pour les rouges)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Convertir notre couleur cible en HSV
        target_pixel = np.uint8([[color]])
        target_hsv = cv2.cvtColor(target_pixel, cv2.COLOR_BGR2HSV)[0][0]
        
        # Définir les plages pour cette couleur en HSV
        h_value = target_hsv[0]
        s_value = target_hsv[1]
        v_value = target_hsv[2]
        
        # Plage HSV adaptée au rouge
        if h_value < 15 or h_value > 165:  # Autour de 0/180 (rouge)
            # Premier masque: rouges basses valeurs H
            lower_hsv1 = np.array([0, max(0, s_value - 50), max(0, v_value - 50)])
            upper_hsv1 = np.array([15, 255, 255])
            mask_hsv1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
            
            # Deuxième masque: rouges hautes valeurs H
            lower_hsv2 = np.array([165, max(0, s_value - 50), max(0, v_value - 50)])
            upper_hsv2 = np.array([180, 255, 255])
            mask_hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
            
            # Combiner les deux masques
            mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
        else:
            # Pour les autres couleurs
            lower_hsv = np.array([max(0, h_value - 10), max(0, s_value - 50), max(0, v_value - 50)])
            upper_hsv = np.array([min(180, h_value + 10), 255, 255])
            mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # Combiner les deux approches
        color_mask = cv2.bitwise_or(color_mask, mask_hsv)
        
        # Ajouter cette couleur au masque global
        mask = cv2.bitwise_or(mask, color_mask)
    
    # Nettoyer le masque avec des opérations morphologiques
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Éliminer les très petits objets (bruit)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10:  # Ignorer les très petits objets
            cv2.drawContours(mask, [contour], -1, 0, -1)
    
    # Afficher les statistiques du masque
    white_pixels = np.sum(mask > 0)
    percentage = (white_pixels / (height * width)) * 100
    print(f"Masque: {white_pixels} pixels détectés ({percentage:.4f}% de l'image)")
    
    # Si aucun pixel n'est détecté, essayer avec une tolérance plus élevée
    if white_pixels < 10 and tolerance < 100:
        print(f"Très peu de pixels détectés. Essai avec tolérance={tolerance*1.5}")
        return create_mask_from_colors(image_path, output_path, colors, int(tolerance*1.5))
    
    # Sauvegarder le masque
    cv2.imwrite(output_path, mask)
    
    return True

def process_all_images_adaptive(images_dir, output_dir, base_color="#aa0003", tolerance=30):
    """
    Traite toutes les images en détectant les couleurs d'annotation spécifiques à chaque image.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Lister tous les fichiers d'images
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        input_path = os.path.join(images_dir, img_file)
        
        # Créer un nom pour le masque
        mask_name = img_file.replace('.png', '_mask.png')
        output_path = os.path.join(output_dir, mask_name)
        
        print(f"\nTraitement de {img_file}...")
        
        # Détecter les couleurs d'annotation dans cette image
        annotation_colors = detect_annotation_colors(input_path, base_color)
        
        if not annotation_colors:
            print(f"Aucune couleur d'annotation trouvée dans {img_file}")
            continue
        
        # Créer le masque avec ces couleurs
        if create_mask_from_colors(input_path, output_path, annotation_colors, tolerance):
            print(f"Masque créé: {output_path}")
        else:
            print(f"Échec de traitement: {input_path}")

def visualize_mask(image_path, mask_path, output_path=None):
    """
    Visualise une image et son masque, avec coloration des carotides en rouge.
    """
    # Charger l'image et le masque
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        print("Impossible de charger l'image ou le masque")
        return
    
    # Convertir l'image en RGB pour l'affichage
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Créer une superposition
    overlay = image_rgb.copy()
    overlay[mask > 0] = [255, 0, 0]  # Rouge pour les carotides
    
    # Mélanger l'image originale et la superposition
    alpha = 0.7
    blended = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)
    
    # Afficher les images
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Image originale')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Masque binaire')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(blended)
    plt.title('Superposition')
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualisation sauvegardée: {output_path}")
    else:
        plt.show()

def process_single_image(image_path, output_path, base_color="#aa0003", tolerance=30):
    """
    Traite une seule image avec une approche adaptative.
    """
    print(f"Traitement de {os.path.basename(image_path)}...")
    
    # Détecter les couleurs d'annotation
    annotation_colors = detect_annotation_colors(image_path, base_color)
    
    if not annotation_colors:
        print("Aucune couleur d'annotation trouvée")
        return False
    
    # Créer le masque
    success = create_mask_from_colors(image_path, output_path, annotation_colors, tolerance)
    
    if success:
        print(f"Masque créé: {output_path}")
    else:
        print("Échec de création du masque")
    
    return success

# Exemple d'utilisation
if __name__ == "__main__":
    # Définir les chemins
    BASE_DIR = "/Users/moneillon/Programmes/TPI-RQCL-DeepBridge/IACarotideU-Net/imagesdata"
    IMAGES_DIR = os.path.join(BASE_DIR, "masks")
    CORRECTED_MASKS_DIR = os.path.join(BASE_DIR, "corrected_masks")
    VISUALIZATION_DIR = os.path.join(BASE_DIR, "mask_verification")
    
    # Créer les répertoires s'ils n'existent pas
    os.makedirs(CORRECTED_MASKS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # Option 1: Traiter toutes les images avec l'approche adaptative
    process_all_images_adaptive(IMAGES_DIR, CORRECTED_MASKS_DIR, "#aa0003", tolerance=30)
    
    # Option 2: Traiter une seule image spécifique
    # image_file = "carotide13.png"
    # input_path = os.path.join(IMAGES_DIR, image_file)
    # output_path = os.path.join(CORRECTED_MASKS_DIR, image_file.replace('.png', '_mask.png'))
    # process_single_image(input_path, output_path, "#aa0003", tolerance=40)
    
    # Visualiser les résultats
    for i in range(10, 20):  # Visualiser les images 10 à 19
        img_file = f"carotide{i}.png"
        img_path = os.path.join(IMAGES_DIR, img_file)
        mask_path = os.path.join(CORRECTED_MASKS_DIR, f"carotide{i}_mask.png")
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            output_path = os.path.join(VISUALIZATION_DIR, f"verify_carotide{i}.png")
            visualize_mask(img_path, mask_path, output_path)
    
    print("Traitement terminé!")
