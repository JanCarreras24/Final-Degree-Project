import pandas as pd

# Cargar el archivo CSV
csv_path = "/home/aih/jan.boada/project/codes/classification/matek_metadata.csv"
metadata = pd.read_csv(csv_path)

# Diccionario para mapear números a nombres
labels_map = {
    0: 'Basophil',
    1: 'Eosinophil',
    2: 'Erythroblast',
    3: 'Atypical Lymphocyte',
    4: 'Typical Lymphocyte',
    5: 'Metamyelocyte',
    6: 'Monoblast',
    7: 'Monocyte',
    8: 'Myeloblast',
    9: 'Myelocyte',
    10: 'Band Neutrophil',
    11: 'Segmented Neutrophil',
    12: 'Promyelocyte',
    13: 'Promyelocyte Bilobed',
    14: 'Smudge cell'
}

# Reemplazar los números por los nombres de las clases
metadata["label"] = metadata["label"].map(labels_map)

# Guardar el archivo modificado
metadata.to_csv(csv_path, index=False)