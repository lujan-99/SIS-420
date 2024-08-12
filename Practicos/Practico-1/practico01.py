import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Fijamos una semilla para asegurar la reproducibilidad de los resultados
np.random.seed(42)

# Generamos 100 estaturas aleatorias entre 1.4m y 2.0m
estaturas = np.random.uniform(1.4, 2.0, 100)

pesos = []  # Lista para almacenar los pesos generados

# Bucle para generar pesos aleatorios controlados según la estatura
for estatura in estaturas:
    # Calcular el peso mínimo y máximo usando el IMC saludable (18.5 a 24.9)
    peso_min = 18.5 * (estatura ** 2)  # Peso mínimo según IMC de 18.5
    peso_max = 24.9 * (estatura ** 2)  # Peso máximo según IMC de 24.9
    # Generar un peso aleatorio entre el peso mínimo y máximo calculado
    peso = np.random.uniform(peso_min, peso_max)
    pesos.append(peso)  # Añadir el peso a la lista de pesos

# Crear un DataFrame con los datos de estatura y peso
datos = pd.DataFrame({
    'Estatura (m)': estaturas,
    'Peso (kg)': pesos
})

# Mostrar los primeros datos generados
print(datos.head())

# Crear una gráfica de estatura vs peso
plt.figure(figsize=(10, 6))
plt.scatter(datos['Estatura (m)'], datos['Peso (kg)'], color='blue', label='Datos generados')
plt.title('Estatura vs Peso')
plt.xlabel('Estatura (m)')
plt.ylabel('Peso (kg)')
plt.legend()
plt.grid(True)
plt.show()

# Guardar el DataFrame en un archivo CSV
datos.to_csv('alturas_pesos_generados.csv', index=False)



# Método 1: Búsqueda Exhaustiva con Rango Adecuado
def busqueda_exhaustiva(x, y):
    min_error = float('inf')
    best_m = None
    best_b = None
    m_range = np.arange(50, 90, 0.1)  # Pendiente variando de 50 a 90
    b_range = np.arange(-70, -60, 0.1)  # Intercepto variando de -70 a -60
    
    for m_float in m_range:
        for b_float in b_range:
            error = np.sum((y - (m_float * x + b_float)) ** 2)
            if error < min_error:
                min_error = error
                best_m = m_float
                best_b = b_float
    return best_m, best_b

x = datos['Estatura (m)']
y = datos['Peso (kg)']

best_m, best_b = busqueda_exhaustiva(x, y)

# Método 2: Regresión Lineal con Fórmulas Directas
m_directo = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
b_directo = np.mean(y) - m_directo * np.mean(x)

# Método 3: Regresión Lineal Usando `scikit-learn`
modelo = LinearRegression()
x_reshaped = x.values.reshape(-1, 1)  # `scikit-learn` requiere que x sea una matriz 2D
modelo.fit(x_reshaped, y)
m_sklearn = modelo.coef_[0]
b_sklearn = modelo.intercept_

# Gráfica 1: Búsqueda Exhaustiva
plt.figure(figsize=(8, 6))
plt.scatter(datos['Estatura (m)'], datos['Peso (kg)'], color='blue', label='Datos generados')
plt.plot(x, best_m * x + best_b, color='red', label='Búsqueda Exhaustiva')
plt.title('Estatura vs Peso - Búsqueda Exhaustiva')
plt.xlabel('Estatura (m)')
plt.ylabel('Peso (kg)')
plt.legend()
plt.grid(True)
plt.show()

# Gráfica 2: Regresión Directa
plt.figure(figsize=(8, 6))
plt.scatter(datos['Estatura (m)'], datos['Peso (kg)'], color='blue', label='Datos generados')
plt.plot(x, m_directo * x + b_directo, color='green', linestyle='--', label='Regresión Directa')
plt.title('Estatura vs Peso - Regresión Directa')
plt.xlabel('Estatura (m)')
plt.ylabel('Peso (kg)')
plt.legend()
plt.grid(True)
plt.show()

# Gráfica 3: Regresión `scikit-learn`
plt.figure(figsize=(8, 6))
plt.scatter(datos['Estatura (m)'], datos['Peso (kg)'], color='blue', label='Datos generados')
plt.plot(x, m_sklearn * x + b_sklearn, color='orange', linestyle=':', label='Regresión `scikit-learn`')
plt.title('Estatura vs Peso - Regresión `scikit-learn`')
plt.xlabel('Estatura (m)')
plt.ylabel('Peso (kg)')
plt.legend()
plt.grid(True)
plt.show()


X = datos['Estatura (m)'].values.reshape(-1, 1)  # Estatura (característica X)
y = datos['Peso (kg)'].values  # Peso (variable objetivo y)

# Calcular el error cuadrático medio para la búsqueda exhaustiva
predicciones_exhaustiva = best_m * X.flatten() + best_b
mse_exhaustiva = mean_squared_error(y, predicciones_exhaustiva)

# Ajuste polinómico de grado 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)  # Genera nuevas características polinómicas basadas en X

model = LinearRegression()
model.fit(X_poly, y)

y_pred_poly = model.predict(X_poly)

# Evaluar el modelo polinómico
mse_poly = mean_squared_error(y, y_pred_poly)
print(f"Error cuadrático medio (MSE) para ajuste polinómico: {mse_poly:.2f}")
print(f"Mejor m (búsqueda exhaustiva): {best_m}")
print(f"Mejor b (búsqueda exhaustiva): {best_b}")
print(f"Error cuadrático medio (búsqueda exhaustiva): {mse_exhaustiva:.2f}")

# Visualización de los resultados
plt.figure(figsize=(12, 6))

# Gráfico de ajuste polinómico
plt.subplot(1, 2, 1)
plt.scatter(datos['Estatura (m)'], datos['Peso (kg)'], color='blue', label='Datos')
plt.plot(np.sort(estaturas), model.predict(poly.transform(np.sort(estaturas).reshape(-1, 1))), color='red', label='Curva ajustada (grado 2)')
plt.title('Curva Ajustada para Estatura vs Peso (Polinómico)')
plt.xlabel('Estatura (m)')
plt.ylabel('Peso (kg)')
plt.legend()

# Gráfico de ajuste con búsqueda exhaustiva
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Datos generados')
plt.plot(X, best_m * X.flatten() + best_b, color='red', linestyle='--', label='Mejor ajuste (Exhaustiva)')
plt.title('Ajuste con Búsqueda Exhaustiva')
plt.xlabel('Estatura (m)')
plt.ylabel('Peso (kg)')
plt.legend()

plt.tight_layout()
plt.show()


# Calcular las predicciones y el error cuadrático medio (MSE) para el modelo de búsqueda exhaustiva
predicciones_exhaustiva = best_m * x + best_b
mse_exhaustiva = np.mean((y - predicciones_exhaustiva) ** 2)

# Seleccionar una muestra de 20 datos para mostrar
muestra = datos.sample(n=20, random_state=42)
muestra['Predicción'] = best_m * muestra['Estatura (m)'] + best_b
muestra['Error'] = abs(muestra['Peso (kg)'] - muestra['Predicción'])

# Imprimir los resultados en formato de tabla
print("Resultados del Modelo de Búsqueda Exhaustiva:")
print(f"Mejor Pendiente (m): {best_m}")
print(f"Mejor Intercepto (b): {best_b}")
print(f"Error Cuadrático Medio (MSE): {mse_exhaustiva:.2f}\n")

print("Muestra de Datos y Errores:")
print(muestra[['Estatura (m)', 'Peso (kg)', 'Predicción', 'Error']])