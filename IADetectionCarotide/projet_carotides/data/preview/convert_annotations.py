import pandas as pd
import os
from pathlib import Path
import json
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def convert_via_to_yolo(csv_path, output_dir):
    """
    Convertit les annotations VIA CSV en format YOLO
    Structure CSV attendue:
    - filename, file_size, region_shape_attributes, etc.
    """
    # Création du dossier de sortie
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Lecture du fichier CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logging.info(f"Nombre total d'annotations: {len(df)}")
    
    # Compteurs pour le suivi
    processed_images = set()
    processed_annotations = 0
    errors = 0
    
    # Traitement des annotations
    for _, row in df.iterrows():
        try:
            filename = row['filename']
            
            # Extraction des attributs de forme
            shape_attributes = json.loads(
                row['region_shape_attributes'].replace("'", '"')
            )
            
            # Vérification du type de forme (rectangle)
            if shape_attributes.get('name') != 'rect':
                logging.warning(f"Forme non rectangulaire trouvée dans {filename}")
                continue
            
            # Extraction des dimensions de l'image
            img_width, img_height = map(int, str(row['file_size']).split(','))
            
            # Extraction des coordonnées du rectangle
            x = float(shape_attributes['x'])
            y = float(shape_attributes['y'])
            width = float(shape_attributes['width'])
            height = float(shape_attributes['height'])
            
            # Conversion en format YOLO (normalisé)
            x_center = (x + width/2) / img_width
            y_center = (y + height/2) / img_height
            width_norm = width / img_width
            height_norm = height / img_height
            
            # Création de la ligne YOLO: <class> <x_center> <y_center> <width> <height>
            yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
            
            # Sauvegarde dans le fichier correspondant
            output_file = output_path / f"{Path(filename).stem}.txt"
            with open(output_file, 'a') as f:
                f.write(yolo_line + '\n')
            
            processed_images.add(filename)
            processed_annotations += 1
            
            # Log de progression
            if processed_annotations % 100 == 0:
                logging.info(f"Annotations traitées: {processed_annotations}")
                
        except Exception as e:
            logging.error(f"Erreur avec {filename}: {str(e)}")
            errors += 1
            continue
    
    # Rapport final
    logging.info("\nConversion terminée !")
    logging.info(f"Images traitées: {len(processed_images)}")
    logging.info(f"Annotations totales: {processed_annotations}")
    logging.info(f"Erreurs: {errors}")
    
    # Création du fichier data.yaml pour YOLO
    yaml_content = {
        'path': str(output_path.parent.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': ['carotide']
    }
    
    import yaml
    with open(output_path.parent / 'data.yaml', 'w') as f:
        yaml.safe_dump(yaml_content, f)

if __name__ == "__main__":
    # Chemins
    base_dir = Path("carotid_project/data")
    csv_path = base_dir / "annotations" / "via_project_csv.csv"
    output_dir = base_dir / "labels"
    
    # Conversion
    convert_via_to_yolo(str(csv_path), str(output_dir))