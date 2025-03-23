import os
import pydicom
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import logging
from dataclasses import dataclass

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class ProcessingConfig:
    """Configuration pour le traitement des images"""
    clahe_clip_limit: float = 1.5
    clahe_grid_size: Tuple[int, int] = (8, 8)
    bilateral_d: int = 7
    bilateral_sigma_color: int = 50
    bilateral_sigma_space: int = 50
    grid_divisions: int = 10
    darkness_factor: float = 0.7  # Facteur d'assombrissement (0-1)

def enhance_image(
    image: np.ndarray,
    config: ProcessingConfig
) -> Optional[np.ndarray]:
    """
    Améliore l'image en préservant la visibilité des carotides
    
    Args:
        image: Image d'entrée en numpy array
        config: Configuration pour le traitement
        
    Returns:
        Image améliorée ou None en cas d'échec
    """
    try:
        # Conversion en float32
        image_float = image.astype(np.float32)
        
        # Normalisation initiale avec étirement du contraste
        p2, p98 = np.percentile(image_float, (2, 98))
        image_norm = (image_float - p2) / (p98 - p2)
        image_norm = np.clip(image_norm, 0, 1)
        
        # Assombrissement de l'image
        image_norm = image_norm * config.darkness_factor
        
        # Conversion en uint8
        image_uint8 = (image_norm * 255).astype(np.uint8)
        
        # Double application de CLAHE pour augmenter le contraste
        clahe = cv2.createCLAHE(
            clipLimit=3.0,  # Augmentation du clipLimit
            tileGridSize=(8, 8)
        )
        image_enhanced = clahe.apply(image_uint8)
        image_enhanced = clahe.apply(image_enhanced)  # Deuxième application
        
        # Application d'une courbe gamma pour accentuer les zones sombres
        gamma = 1.3  # Augmentation du gamma
        image_enhanced = np.power(image_enhanced / 255.0, gamma) * 255.0
        image_enhanced = image_enhanced.astype(np.uint8)
        
        # Amélioration supplémentaire du contraste
        image_enhanced = cv2.equalizeHist(image_enhanced)
        
        # Filtre bilatéral pour préserver les bords tout en réduisant le bruit
        image_smooth = cv2.bilateralFilter(
            image_enhanced,
            d=7,
            sigmaColor=30,  # Réduction pour plus de contraste
            sigmaSpace=30
        )
        
        return image_smooth
        
    except Exception as e:
        logging.error(f"Erreur lors de l'amélioration de l'image: {str(e)}")
        return None
        
    except Exception as e:
        logging.error(f"Erreur lors de l'amélioration de l'image: {str(e)}")
        return None

class DicomProcessor:
    def __init__(
        self,
        input_path: str,
        output_dir: str,
        config: Optional[ProcessingConfig] = None
    ):
        """
        Initialise le processeur DICOM
        
        Args:
            input_path: Chemin vers les fichiers DICOM
            output_dir: Dossier de sortie
            config: Configuration pour le traitement des images
        """
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.config = config or ProcessingConfig()
        
        # Définition des sous-dossiers
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.preview_dir = self.output_dir / "preview"
        self.failed_dir = self.output_dir / "failed"
        
        # Création des dossiers
        for directory in [self.images_dir, self.labels_dir,
                         self.preview_dir, self.failed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Création du fichier classes.txt pour YOLO
        with open(self.output_dir / "classes.txt", "w") as f:
            f.write("carotide\n")
            
    def create_preview_grid(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Crée une image avec grille pour l'annotation
        
        Args:
            image: Image d'entrée
            
        Returns:
            Image avec grille
        """
        preview_image = image.copy()
        h, w = preview_image.shape
        
        # Ajout de la grille
        for x in range(0, w, w // self.config.grid_divisions):
            cv2.line(preview_image, (x, 0), (x, h), (128, 128, 128), 1)
        for y in range(0, h, h // self.config.grid_divisions):
            cv2.line(preview_image, (0, y), (w, y), (128, 128, 128), 1)
            
        return preview_image
        
    def process_single_file(
        self,
        file_path: str,
        base_name: str
    ) -> bool:
        """
        Traite un fichier DICOM individuel
        
        Args:
            file_path: Chemin vers le fichier DICOM
            base_name: Nom de base pour les fichiers de sortie
            
        Returns:
            True si le traitement a réussi, False sinon
        """
        try:
            # Lecture DICOM
            dicom = pydicom.dcmread(file_path)
            image = dicom.pixel_array
            
            # Amélioration de l'image
            enhanced_image = enhance_image(image, self.config)
            if enhanced_image is None:
                raise ValueError("Échec de l'amélioration de l'image")
            
            # Sauvegarde de l'image
            image_path = self.images_dir / f"{base_name}.png"
            Image.fromarray(enhanced_image).save(image_path)
            
            # Création du fichier d'annotation vide
            label_path = self.labels_dir / f"{base_name}.txt"
            label_path.touch()
            
            # Création et sauvegarde de la preview avec grille
            preview_image = self.create_preview_grid(enhanced_image)
            cv2.imwrite(
                str(self.preview_dir / f"{base_name}_grid.png"),
                preview_image
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Erreur avec {file_path}: {str(e)}")
            # Copie du fichier problématique dans le dossier failed
            dest = self.failed_dir / Path(file_path).name
            try:
                import shutil
                shutil.copy2(file_path, dest)
            except Exception as copy_error:
                logging.error(f"Impossible de copier le fichier échoué: {str(copy_error)}")
            return False

    def process_directory(self) -> Tuple[int, int]:
        """
        Traite tous les fichiers DICOM du répertoire
        
        Returns:
            Tuple contenant le nombre de fichiers traités et le nombre d'erreurs
        """
        processed = 0
        errors = 0
        
        for root, _, files in os.walk(self.input_path):
            for file in files:
                file_path = os.path.join(root, file)
                base_name = f"image_{processed:05d}"
                
                if self.process_single_file(file_path, base_name):
                    processed += 1
                    if processed % 10 == 0:
                        logging.info(f"Traités: {processed} fichiers")
                else:
                    errors += 1
        
        return processed, errors

def validate_dicom_directory(path: str) -> bool:
    """
    Vérifie si le dossier contient des fichiers DICOM valides
    """
    try:
        # Conversion en chemin absolu et normalisation
        path = str(Path(path).resolve())
        if len(path) > 255 and os.name == 'nt':  # Windows
            path = '\\\\?\\' + path  # Préfixe pour les longs chemins Windows
            
        if not os.path.exists(path):
            logging.error(f"Le dossier {path} n'existe pas")
            return False
            
        # Vérifie s'il y a au moins un fichier DICOM valide
        found_dicom = False
        for entry in os.scandir(path):
            try:
                if entry.is_file():
                    dicom = pydicom.dcmread(str(entry.path), force=True)
                    if hasattr(dicom, 'pixel_array'):
                        found_dicom = True
                        logging.info(f"Fichier DICOM valide trouvé : {entry.name}")
                        break
            except Exception as e:
                logging.debug(f"Fichier ignoré {entry.name}: {str(e)}")
                continue
                
        if not found_dicom:
            logging.error(f"Aucun fichier DICOM valide trouvé dans {path}")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Erreur lors de la validation du dossier: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    import os
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Création du parser d'arguments
    parser = argparse.ArgumentParser(description='Traitement des images DICOM pour la détection des carotides')
    parser.add_argument('input_path', help='Chemin vers le dossier contenant les fichiers DICOM')
    parser.add_argument('--output-dir', default='carotid_project/data',
                      help='Dossier de sortie (défaut: carotid_project/data)')
    parser.add_argument('--darkness', type=float, default=0.5,  # Changé à 0.5 par défaut
                      help='Facteur d\'assombrissement (0-1, défaut: 0.5)')
    
    args = parser.parse_args()
    
    # Conversion du chemin en absolu et normalisation
    input_path = str(Path(args.input_path).resolve())
    output_dir = str(Path(args.output_dir).resolve())
    
    if len(input_path) > 255 and os.name == 'nt':  # Windows
        input_path = '\\\\?\\' + input_path  # Préfixe pour les longs chemins Windows
    
    logging.info(f"Dossier d'entrée : {input_path}")
    logging.info(f"Dossier de sortie : {output_dir}")
    
    # Validation du dossier d'entrée
    if not validate_dicom_directory(input_path):
        logging.error("Impossible de continuer sans dossier DICOM valide")
        exit(1)
        
    # Configuration
    config = ProcessingConfig(
        darkness_factor=args.darkness,
        clahe_clip_limit=3.0,  # Augmenté pour plus de contraste
        clahe_grid_size=(8, 8),
        bilateral_d=7,
        bilateral_sigma_color=30,  # Réduit pour plus de contraste
        bilateral_sigma_space=30,  # Réduit pour plus de contraste
        grid_divisions=12
    )
    
    # Création et exécution du processeur
    try:
        processor = DicomProcessor(input_path, output_dir, config)
        logging.info("Début du traitement des images DICOM...")
        processed, errors = processor.process_directory()
        
        logging.info(f"\nTraitement terminé !")
        logging.info(f"Fichiers traités : {processed}")
        logging.info(f"Erreurs : {errors}")
        
        if processed > 0:
            logging.info(f"\nLes images sont prêtes pour l'annotation dans : {output_dir}/images")
            logging.info(f"Une grille d'aide à l'annotation est disponible dans : {output_dir}/preview")
        if errors > 0:
            logging.info(f"Les fichiers problématiques ont été copiés dans : {output_dir}/failed")
            
    except Exception as e:
        logging.error(f"Erreur fatale : {str(e)}")
        exit(1)
    
    # Création et exécution du processeur
    try:
        processor = DicomProcessor(input_path, args.output_dir, config)
        logging.info("Début du traitement des images DICOM...")
        processed, errors = processor.process_directory()
        
        logging.info(f"\nTraitement terminé !")
        logging.info(f"Fichiers traités : {processed}")
        logging.info(f"Erreurs : {errors}")
        
        if processed > 0:
            logging.info(f"\nLes images sont prêtes pour l'annotation dans : {args.output_dir}/images")
            logging.info(f"Une grille d'aide à l'annotation est disponible dans : {args.output_dir}/preview")
        if errors > 0:
            logging.info(f"Les fichiers problématiques ont été copiés dans : {args.output_dir}/failed")
            
    except Exception as e:
        logging.error(f"Erreur fatale : {str(e)}")
        exit(1)
    
    # Configuration personnalisée (optionnelle)
    config = ProcessingConfig(
        clahe_clip_limit=2.0,  # Augmentation légère du contraste
        clahe_grid_size=(16, 16),  # Grille plus fine pour CLAHE
        bilateral_d=9,  # Augmentation légère du lissage
        grid_divisions=12,  # Plus de divisions dans la grille d'annotation
        darkness_factor=0.7  # Assombrissement de l'image (0.7 = 30% plus sombre)
    )
    
    # Création et exécution du processeur
    processor = DicomProcessor(input_path, output_dir, config)
    
    logging.info("Début du traitement des images DICOM...")
    processed, errors = processor.process_directory()
    
    logging.info(f"\nTraitement terminé !")
    logging.info(f"Fichiers traités : {processed}")
    logging.info(f"Erreurs : {errors}")
    logging.info(f"\nLes images sont prêtes pour l'annotation dans : {output_dir}/images")
    logging.info(f"Une grille d'aide à l'annotation est disponible dans : {output_dir}/preview")
    if errors > 0:
        logging.info(f"Les fichiers problématiques ont été copiés dans : {output_dir}/failed")