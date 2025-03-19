from ultralytics import YOLO
import os

# Chemin vers votre fichier data.yaml
yaml_path = r"C:\dataset_chu_nice_2020_2021\scan\SF103E8_10.241.3.232_20210118173900817_CT\SF103E8_10.241.3.232_20210118173900817\1.2.840.113619.2.55.3.2148147470.648.1353479279.648\carotid_project\data\data.yaml"

def train_yolo():
    # Charger un modèle pré-entraîné
    model = YOLO('yolov8n.pt')  # 'n' pour nano (plus petit et plus rapide)
    
    # Lancer l'entraînement
    results = model.train(
        data=yaml_path,          # chemin vers votre fichier data.yaml
        epochs=100,              # nombre d'époques
        imgsz=640,              # taille des images
        batch=16,               # taille du batch
        patience=20,            # early stopping
        device='0' if os.name != 'nt' else 'cpu',  # GPU si disponible, sinon CPU
        project='carotid_detection',
        name='train2'
    )
    
    # Valider le modèle sur l'ensemble de test
    model.val()

if __name__ == "__main__":
    train_yolo()