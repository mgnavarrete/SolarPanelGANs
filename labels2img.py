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
print("1. Seleccione el directorio raíz de las imágenes")
print("2. Seleccione el directorio raíz de las etiquetas")
print("3. Seleccione los directorios de las imágenes copiar")
print("4. Espere a que el proceso termine")
print("5. ¡Listo!")

# Seleccionar directorios de imágenes y etiquetas
imagesPath = filedialog.askdirectory(title='Seleccione el directorio raíz de images')
labelsPath = filedialog.askdirectory(title='Seleccione el directorio raíz de labels')
# Guarda en lista los archivos en labelsPath
labels = os.listdir(labelsPath)
labelsClean = []
for label in tqdm(labels, desc="Recopilando nombres de labels"):
    labelsClean.append(label.split(".")[0])

listPath = select_directories()
for path_root in listPath:
    for folder_path in os.listdir(path_root):
        if folder_path.endswith('PP'):
            # Recorrer todos los archivos en la carpeta
            path = os.path.join(path_root, folder_path)
            
            for filename in tqdm(os.listdir(os.path.join(path,"cvat")),desc="Contando Imágenes"):
                # Verifica si el nombre de archivo (sin la extensión) está en labelsClean
                label_name = filename.split(".")[0]
                if label_name in labelsClean:
                    # Agrega el prefijo "F_" tanto al label como al nombre de archivo
                    new_label_name = "F_" + label_name
                    new_filename = "F_" + filename
                    
                    # Construir la ruta completa al archivo original y la ruta de destino con el nuevo nombre
                    file_path = os.path.join(path, "cvat", filename)
                    destination_path = os.path.join(imagesPath, new_filename)
                    
                    # Copia el archivo con el nuevo nombre al directorio 'images'
                    shutil.copy(file_path, destination_path)
                    # Cambiar nombre a label_name
                    os.rename(os.path.join(labelsPath, label_name + ".txt"), os.path.join(labelsPath, new_label_name + ".txt"))

                        
