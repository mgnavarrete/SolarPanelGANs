import os
import numpy as np

labels_path = "C:/Users/Adentu/Desktop/allData/labels"
num_classes = 8  # Número de tipos de fallas

image_conditions = {}

for label_file in os.listdir(labels_path):
    file_path = os.path.join(labels_path, label_file)
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Si el archivo esta vacio
        if not lines:
            condition_vector = np.zeros(num_classes)
        # Inicializa el vector de condición como un vector de ceros
        else:
            condition_vector = np.zeros(num_classes)
            for line in lines:
                # Si la linea esta vacio
                if not line.strip():
                    continue
                class_id, x_center, y_center, width, height = map(float, line.split())
                # Incrementa el contador correspondiente al class_id en el vector de condición
                condition_vector[int(class_id)] += 1
        
        # Guarda el vector de condición para esta imagen en un archivo .npy
        npy_file = label_file.replace('.txt', '.npy')
        np.save(os.path.join(labels_path, npy_file), condition_vector)        
