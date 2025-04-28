import os
import pandas as pd
import random
import numpy as np

# Directorio donde se encuentran las imágenes
data_dir = '/home/aih/jan.boada/project/codes/datasets/data/matek/real_train_fewshot/seed0'

# Mapa de etiquetas que has proporcionado
labels_map = {
    'Basophil': 0,
    'Eosinophil': 1,
    'Erythroblast': 2,
    'Atypical Lymphocyte': 3,
    'Typical Lymphocyte': 4,
    'Metamyelocyte': 5,
    'Monoblast': 6,
    'Monocyte': 7, 
    'Myeloblast': 8,
    'Myelocyte': 9,
    'Band Neutrophil': 10,
    'Segmented Neutrophil': 11, 
    'Promyelocyte': 12,
    'Promyelocyte Bilobed': 13,
    'Smudge cell': 14
}

# Lista para almacenar los datos
data = []

# Recorremos las carpetas que contienen las imágenes
for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)
    
    # Si es un directorio, procesamos las imágenes dentro de esa carpeta
    if os.path.isdir(folder_path):
        # Obtenemos la etiqueta correspondiente según el nombre de la carpeta
        label = labels_map.get(folder_name)
        
        # Recorremos todas las imágenes dentro de la carpeta
        images = [img_name for img_name in os.listdir(folder_path) if img_name.endswith(('.tiff', '.jpg', '.png', '.jpeg'))]
        
        # Contamos el número de imágenes
        num_images = len(images)
        num_folds = 5
        
        # Si el número de imágenes no es divisible por 5, seleccionamos una imagen aleatoria para completarlo
        if num_images % num_folds != 0:
            missing_images = num_folds - (num_images % num_folds)
            for _ in range(missing_images):
                images.append(random.choice(images))  # Añadimos una imagen aleatoria para completar los folds

        # Mezclamos las imágenes aleatoriamente
        random.shuffle(images)

        # Dividimos las imágenes en 5 folds
        fold_size = len(images) // num_folds
        folds = [images[i:i + fold_size] for i in range(0, len(images), fold_size)]

        # Nos aseguramos de que la lista de folds tenga exactamente 5 elementos
        while len(folds) < num_folds:
            folds.append([])

        # Recorremos los folds para asignar 'train' y 'test'
        for fold_idx in range(num_folds):
            set_values = ['test' if i == fold_idx else 'train' for i in range(num_folds)]
            
            for img_name in folds[fold_idx]:
                img_path = os.path.join(folder_path, img_name)
                
                # Asignar dataset
                dataset_name = 'matek'
                
                # Añadimos la imagen, la etiqueta, el dataset y los valores de los folds
                data.append([img_path, label, dataset_name] + set_values)

# Creamos un DataFrame de pandas con los datos recopilados
df = pd.DataFrame(data, columns=['image', 'label', 'dataset', 'set0', 'set1', 'set2', 'set3', 'set4'])

# Guardamos el DataFrame como un archivo CSV
csv_file = '/home/aih/jan.boada/project/codes/classification/matek_metadata.csv'
df.to_csv(csv_file, index=False)

print(f"CSV creado con éxito: {csv_file}")
