import os
import shutil
from tkinter import filedialog
from tqdm import tqdm


def select_directories():
    list_folders = []  
    path_root = filedialog.askdirectory(title='Seleccione el directorio raíz')
    while path_root:
        list_folders.append(path_root)
        path_root = filedialog.askdirectory(title='Seleccione otro directorio o cancele para continuar')
    if not list_folders:
        raise Exception("No se seleccionó ningún directorio")
    return list_folders

print("Paso a paso:")
print("1. Seleccionar origen")
print("2. Seleccione donde copiar")


# Seleccionar directorios de imágenes y etiquetas
originPath = filedialog.askdirectory(title='Seleccione el directorio raíz de images')
toCopyPath = filedialog.askdirectory(title='Seleccione el directorio raíz de labels')
# Guarda en lista los archivos en labelsPath
originLabels = os.path.join(originPath, "labels")
originImages = os.path.join(originPath, "images")
toCopyLabels = os.path.join(toCopyPath, "labels")
toCopyImages = os.path.join(toCopyPath, "images")

labels = os.listdir(originLabels)
labelsClean = []
for label in tqdm(labels, desc="Recopilando nombres de labels"):
    labelsClean.append(label.split(".")[0])

for filename in tqdm(labelsClean, desc="Copiando Imágenes"):
        
    shutil.copy(os.path.join(originLabels, filename + ".txt"), toCopyLabels)
    shutil.copy(os.path.join(originImages, filename + ".JPG"), toCopyImages)
   
            
