import os
import pydicom
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import logging
from datetime import datetime

# Chemins à personnaliser
BASE_DIR = Path(r"C:\Users\dylan\Documents\SF103E8_10.241.3.232_20210118173228207\1.2.840.113619.2.359.3.1695209168.411.1506489095.530")
MODEL_PATH = r"C:\dataset_chu_nice_2020_2021\scan\SF103E8_10.241.3.232_20210118173900817_CT\SF103E8_10.241.3.232_20210118173900817\1.2.840.113619.2.55.3.2148147470.648.1353479279.648\carotid_detection\train2\weights\best.pt"
OUTPUT_DIR = Path(r"C:\dataset_chu_nice_2020_2021\scan\SF103E8_10.241.3.232_20210118173900817_CT\SF103E8_10.241.3.232_20210118173900817\1.2.840.113619.2.55.3.2148147470.648.1353479279.648\resultats_detection2") / f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

class CarotidDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(OUTPUT_DIR / "detection_log.txt"),
                logging.StreamHandler()
            ]
        )
        
        # Créer le dossier de sortie
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
    def enhance_image(self, image):
        """Améliore l'image DICOM pour une meilleure détection des carotides"""
        image_float = image.astype(np.float32)
        
        if image_float.max() != image_float.min():
            image_norm = (image_float - image_float.min()) / (image_float.max() - image_float.min())
        else:
            return np.zeros_like(image)
        
        # Amélioration plus agressive du contraste
        image_uint8 = (image_norm * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image_enhanced = clahe.apply(image_uint8)
        
        # Réduction du bruit
        image_denoised = cv2.medianBlur(image_enhanced, 3)
        
        return image_denoised

    def process_dicom(self, dicom_path, save_dir):
        try:
            # Lecture et prétraitement
            dicom = pydicom.dcmread(dicom_path)
            image = dicom.pixel_array
            enhanced = self.enhance_image(image)
            rgb_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            
            # Plusieurs passes de détection avec différents seuils
            results = []
            for conf_threshold in [0.25, 0.15, 0.1]:  # Tests avec différents seuils
                result = self.model.predict(
                    source=rgb_image,
                    conf=conf_threshold,
                    verbose=False,
                    iou=0.3  # Seuil IOU plus bas pour détecter des boîtes proches
                )[0]
                
                if len(result.boxes) > 0:
                    results.append(result)
                    if len(result.boxes) >= 2:  # Si on trouve deux carotides, on arrête
                        break
            
            if not results:
                return []
                
            # Utiliser le meilleur résultat (celui avec le plus de détections)
            result = max(results, key=lambda x: len(x.boxes))
            
            # Sauvegarder l'image avec les détections
            annotated_img = result.plot()
            save_path = Path(save_dir) / f"{Path(dicom_path).stem}_detected.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
            
            # Extraire et trier les détections
            detections = []
            boxes = result.boxes
            for box in boxes:
                coords = box.xyxy[0].tolist()
                conf = float(box.conf)
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                detections.append({
                    'coordinates': coords,
                    'confidence': conf,
                    'class': class_name,
                    'image_path': str(save_path)
                })
            
            # Vérifier si on a trouvé moins de 2 carotides
            if len(detections) < 2:
                logging.warning(f"Attention: Seulement {len(detections)} carotide(s) trouvée(s) dans {dicom_path}")
            
            return detections
            
        except Exception as e:
            logging.error(f"Erreur lors du traitement de {dicom_path}: {str(e)}")
            return None

    def process_directory(self, input_dir, output_dir):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        all_results = {}
        processed = 0
        errors = 0
        single_detections = 0
        
        for file_path in input_path.rglob('*'):
            if not file_path.is_file():
                continue
                
            try:
                # Vérifier si c'est un DICOM
                dicom = pydicom.dcmread(str(file_path), force=True)
                if not hasattr(dicom, 'pixel_array'):
                    continue
                
                # Traiter le fichier
                result = self.process_dicom(str(file_path), output_dir)
                if result:
                    all_results[str(file_path)] = result
                    processed += 1
                    
                    if len(result) == 1:
                        single_detections += 1
                    
                if processed % 10 == 0:
                    logging.info(f"Traités: {processed} fichiers")
                    
            except Exception as e:
                logging.error(f"Erreur avec {file_path}: {str(e)}")
                errors += 1
                continue
        
        # Sauvegarder un résumé détaillé
        with open(output_path / "detection_summary.txt", "w") as f:
            f.write(f"Résumé des détections:\n")
            f.write(f"====================\n")
            f.write(f"Fichiers traités: {processed}\n")
            f.write(f"Erreurs: {errors}\n")
            f.write(f"Détections uniques: {single_detections}\n")
            f.write(f"Détections doubles: {processed - single_detections}\n\n")
            
            for file_path, detections in all_results.items():
                f.write(f"\nFichier: {file_path}\n")
                f.write(f"Nombre de carotides détectées: {len(detections)}\n")
                for i, det in enumerate(detections, 1):
                    f.write(f"  Carotide {i}:\n")
                    f.write(f"    Type: {det['class']}\n")
                    f.write(f"    Confiance: {det['confidence']:.3f}\n")
                    f.write(f"    Coordonnées: {det['coordinates']}\n")
        
        logging.info(f"\nTraitement terminé!")
        logging.info(f"Fichiers traités: {processed}")
        logging.info(f"Erreurs: {errors}")
        logging.info(f"Détections uniques: {single_detections}")
        logging.info(f"Détections doubles: {processed - single_detections}")
        logging.info(f"Résultats sauvegardés dans: {output_path}")
        
        return all_results

if __name__ == "__main__":
    detector = CarotidDetector(MODEL_PATH)
    results = detector.process_directory(BASE_DIR, OUTPUT_DIR)