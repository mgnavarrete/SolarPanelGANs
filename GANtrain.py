import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Concatenate, BatchNormalization, LeakyReLU, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt

import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Concatenate, BatchNormalization, LeakyReLU, Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt

def load_data(image_dir, label_dir, image_size=(640, 512)):
    images = []
    labels = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = load_img(img_path, target_size=image_size, color_mode='grayscale')
        img = img_to_array(img) / 255.0
        images.append(img)
        label_name = img_name.replace('.png', '.npy')
        label_path = os.path.join(label_dir, label_name)
        label = np.load(label_path)
        labels.append(label)
    return np.array(images), np.array(labels)

def build_generator(latent_dim, img_shape, conditions_shape):
    noise_input = layers.Input(shape=(latent_dim,))
    condition_input = layers.Input(shape=(conditions_shape,))
    merged_input = layers.Concatenate()([noise_input, condition_input])
    # Ajuste de dimensiones para asegurar alineación con salida deseada
    x = layers.Dense(512 * 8 * 8)(merged_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((8, 8, 512))(x)
    x = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    img_output = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="tanh")(x)
    return models.Model([noise_input, condition_input], img_output)

def build_discriminator(img_shape, conditions_shape):
    img_input = layers.Input(shape=img_shape)
    condition_input = layers.Input(shape=(conditions_shape,))
    flat_img = layers.Flatten()(img_input)
    merged_input = layers.Concatenate()([flat_img, condition_input])
    x = layers.Dense(512)(merged_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.4)(x)  # Añadir Dropout para regularización
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.4)(x)  # Añadir Dropout para regularización
    validity_output = layers.Dense(1, activation="sigmoid")(x)
    return models.Model([img_input, condition_input], validity_output)

# Cargar datos
image_dir = 'K:/dataGAN/images'  
label_dir = 'K:/dataGAN/labels'  
images, labels = load_data(image_dir, label_dir)

# Asegúrate de ajustar estas dimensiones según tus datos y estructura de red
img_shape = (512, 640, 1)
conditions_shape = (8,)
latent_dim = 100

# Construir y compilar el discriminador
discriminator = build_discriminator(img_shape, conditions_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Construir el generador
generator = build_generator(latent_dim, img_shape, conditions_shape)

# El generador toma ruido y la condición como entrada y genera imágenes
noise = Input(shape=(latent_dim,))
condition = Input(shape=(conditions_shape,))
img = generator([noise, condition])

# Para el modelo combinado, solo entrenamos el generador
discriminator.trainable = False

# El discriminador toma las imágenes generadas y la condición como entrada y determina validez
valid = discriminator([img, condition])

# El modelo combinado (stacked generator and discriminator)
combined = Model([noise, condition], valid)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

def train(generator, discriminator, combined, images, labels, epochs, batch_size=32, save_interval=50):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # ---------------------
        #  Entrenar el Discriminador
        # ---------------------
        idx = np.random.randint(0, images.shape[0], batch_size)
        imgs, labels_batch = images[idx], labels[idx]
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict([noise, labels_batch])

        # Entrenamiento del discriminador
        d_loss_real = discriminator.train_on_batch([imgs, labels_batch], valid)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels_batch], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Entrenar el Generador
        # ---------------------
        sampled_labels = np.random.randint(0, labels.max()+1, batch_size)  # Asegúrate de que esto coincida con tus etiquetas
        g_loss = combined.train_on_batch([noise, sampled_labels], valid)

        # Progreso del entrenamiento
        print(f"{epoch} [D loss: {d_loss[0]} - D acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")
        
        # Guardar imágenes generadas cada 'save_interval' épocas
        if epoch % save_interval == 0:
            save_imgs(generator, epoch)

# Función para generar y guardar imágenes tras cierto número de épocas
def save_imgs(generator, epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    sampled_labels = np.random.randint(0, 8, (r * c, labels.shape[1]))  # Ajusta esto para tus datos
    gen_imgs = generator.predict([noise, sampled_labels])

    # Escalado de imágenes 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(f"images/epoch_{epoch}.JPG")
    plt.close()

# Ejecutar el entrenamiento
train(generator, discriminator, combined, images, labels, epochs=10000, batch_size=32, save_interval=200)