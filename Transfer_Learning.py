# importando bibliotecas

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# baixar a base de dados

import tensorflow_datasets as tfds

# Carregar dataset de gatos e cães
(ds_train, ds_val), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
)

# pre-processamento das imagens

IMG_SIZE = 160  # tamanho padrão para MobileNetV2

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0  # normalização
    return image, label

train = ds_train.map(preprocess).batch(32).shuffle(1000)
val = ds_val.map(preprocess).batch(32)

# carregar modelo pré-treinado (MobileNetV2)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Congelar pesos da base

# construir a rede final

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')  # saída binária: cão ou gato
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# treinar o modelo

history = model.fit(
    train,
    validation_data=val,
    epochs=5
)

