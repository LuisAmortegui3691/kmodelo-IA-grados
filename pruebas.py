import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46.4, 59, 71.6, 100.4], dtype=float)

oculta1 = tf.keras.layers.Dense(units=1, input_shape=(1,))
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss="mean_squared_error"
)

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado")

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de la perdida")
plt.plot(historial.history["loss"])  # Corrección en la visualización del historial

print("Hagamos una predicción")
resultado = modelo.predict([80,])
print("El resultado es " + str(resultado) + " Fahrenheit")

print("Variables internas del modelo")
print(oculta1.get_weights())  # Corrección en la impresión de las variables internas


