import pandas as pd
import os
from pathlib import Path
import json
import PIL.Image
import logging
import shutil
import random

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Chemins absolus basés sur votre structure
BASE_DIR = Path(r"C:\dataset_chu_nice_2020_2021\scan\SF103E8_10.241.3.232_20210118173900817_CT\SF103E8_10.241.3.232_20210118173900817\1.2.840.113619.2.55.3.2148147470.648.1353479279.648\carotid_project\data")
IMAGES_DIR = BASE_DIR / "preview"
LABELS_DIR = BASE_DIR / "labels"
CSV_PATH = BASE_DIR / "annotations" / "via_project_csv.csv"

def get_image_size(image_path):
    """Obtenir les dimensions réelles de l'image"""
    try:
        with PIL.Image.open(image_path) as img:
            return img.size
    except Exception as e:
        logging.error(f"Erreur lors de la lecture de {image_path}: {str(e)}")
        return None

def prepare_yolo_folders():
    """Prépare la structure de dossiers pour YOLO"""
    folders = {
        'train': ['images/train', 'labels/train'],
        'val': ['images/val', 'labels/val'],
        'test': ['images/test', 'labels/test']
    }
    
    for split_folders in folders.values():
        for folder in split_folders:
            (BASE_DIR / folder).mkdir(parents=True, exist_ok=True)

def convert_via_to_yolo():
    """Convertit les annotations VIA vers le format YOLO"""
    try:
        # Lecture du CSV
        df = pd.read_csv(CSV_PATH)
        logging.info(f"CSV chargé avec succès: {len(df)} annotations")
        
        # Préparation des dossiers
        prepare_yolo_folders()
        
        # Traitement des annotations
        processed = 0
        annotations_count = 0
        image_annotations = {}
        
        # Mapping des carotides aux classes YOLO
        carotide_mapping = {
            'carotide gauche': 0,
            'carotide droite': 1
        }
        
        for _, row in df.iterrows():
            filename = row['filename']
            image_path = IMAGES_DIR / filename
            
            # Vérifier si l'image existe
            if not image_path.exists():
                logging.error(f"Image non trouvée: {filename}")
                continue
            
            # Obtenir les dimensions de l'image
            size = get_image_size(image_path)
            if not size:
                continue
                
            img_width, img_height = size
            
            # Parser les attributs de région
            region_attributes = json.loads(row['region_attributes'].replace("'", '"'))
            region_shape = json.loads(row['region_shape_attributes'].replace("'", '"'))
            
            # Extraire et normaliser les coordonnées
            x = float(region_shape['x'])
            y = float(region_shape['y'])
            width = float(region_shape['width'])
            height = float(region_shape['height'])
            
            # Conversion au format YOLO
            x_center = (x + width/2) / img_width
            y_center = (y + height/2) / img_height
            width_norm = width / img_width
            height_norm = height / img_height
            
            # Déterminer la classe
            carotide_type = region_attributes.get('carotide', '')
            class_id = carotide_mapping.get(carotide_type, -1)
            
            if class_id == -1:
                logging.warning(f"Classe non reconnue pour {filename}: {carotide_type}")
                continue
            
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
            
            if filename not in image_annotations:
                image_annotations[filename] = []
            image_annotations[filename].append(yolo_line)
            
            annotations_count += 1
        
        # Division des données (70% train, 15% val, 15% test)
        images = list(image_annotations.keys())
        random.shuffle(images)
        
        n_train = int(0.7 * len(images))
        n_val = int(0.15 * len(images))
        
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train+n_val],
            'test': images[n_train+n_val:]
        }
        
        # Déplacer les images et créer les annotations
        for split, split_images in splits.items():
            for filename in split_images:
                # Copier l'image
                src_image = IMAGES_DIR / filename
                dst_image = BASE_DIR / f"images/{split}" / filename
                shutil.copy2(src_image, dst_image)
                
                # Créer l'annotation
                txt_filename = f"{Path(filename).stem}.txt"
                with open(BASE_DIR / f"labels/{split}" / txt_filename, 'w') as f:
                    f.write('\n'.join(image_annotations[filename]))
                
                processed += 1
                
                if processed % 100 == 0:
                    logging.info(f"Traités: {processed} images")
        
        logging.info("\nConversion terminée avec succès!")
        logging.info(f"Images traitées: {processed}")
        logging.info(f"Annotations totales: {annotations_count}")
        logging.info(f"Train: {len(splits['train'])} images")
        logging.info(f"Validation: {len(splits['val'])} images")
        logging.info(f"Test: {len(splits['test'])} images")
        
    except Exception as e:
        logging.error(f"Erreur globale: {str(e)}")

if __name__ == "__main__":
    convert_via_to_yolo()