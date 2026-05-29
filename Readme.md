# Vektra: 3D Generation Engine 🚀

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Vektra** es un motor de generación de mallas 3D que transforma datos abstractos (comandos, funciones matemáticas e imágenes) en objetos tridimensionales interactivos.

---

## 🛠️ Tecnologías Core

- **Procesamiento:** `NumPy` & `SciPy` (Cálculo vectorial y Triangulación).
- **Visión Artificial:** `OpenCV` (Detección de contornos mediante Canny).
- **Visualización:** `Plotly` (Renderizado interactivo `Mesh3D`).

---

## ⚙️ Guía de Instalación Quick-Start

> **Prerequisito:** Tener instalado [Python 3.10+](https://www.python.org/downloads/) y [Git](https://git-scm.com/).

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/vektra.git
cd vektra
```

---

### Paso 2 — Crear el entorno virtual

> Esto aísla las dependencias del proyecto del sistema y evita conflictos.

**🐧 Linux / macOS**

```bash
python3 -m venv .venv
```

**🪟 Windows (PowerShell)**

```powershell
py -m venv .venv
```

---

### Paso 3 — Activar el entorno virtual

> ⚠️ **Debes activarlo cada vez que abras una nueva terminal.**

**🐧 Linux / macOS**

```bash
source .venv/bin/activate
```

**🪟 Windows (PowerShell)**

```powershell
.venv\Scripts\activate
```

Cuando esté activo, verás el prefijo `(.venv)` en tu terminal.

---

### Paso 4 — Instalar dependencias

```bash
pip install -r requirements.txt
```

---

### Paso 5 — Ejecutar la aplicación

```bash
streamlit run app.py
```

La app se abre automáticamente en `http://localhost:8501`.

---

### ▶️ Ejecución permanente (uso diario)

Una vez instalado, para correr el proyecto en cualquier momento solo necesitás:

**🐧 Linux / macOS**

```bash
source .venv/bin/activate && streamlit run app.py
```

**🪟 Windows (PowerShell)**

```powershell
.venv\Scripts\activate; streamlit run app.py
```

---

![img.png](img.png)

---

## 📖 Guía de Uso

1. **Generación Paramétrica (Fórmulas)**

El motor evalúa funciones f(u, v) en tiempo real.

Ejemplo: La "Silla de Montar" se genera procesando la función z = u^2 - v^2 mediante el módulo _parametric.py_.

2. **Extrusión desde Imagen**

Transforma una fotografía en un sólido 3D detectando sus bordes exteriores. El pipeline sigue este orden:

1. Lectura de imagen (ej: image_f258a0.jpg).
2. Conversión a escala de grises y desenfoque Gaussiano.
3. Detección de bordes con Canny.
4. Generación de malla 3D mediante extrusión de los puntos detectados.

5. **Funciones Sonoras (Síntesis de Fourier)**

Visualiza la construcción procedural de señales de audio complejas mediante la superposición armónica de ondas sinusoidales.
Permite evaluar interactivamente el impacto del número de armónicos (N) en el modelado geométrico y topológico de 6 ondas del catálogo clásico:

1. **Onda Cuadrada:** Sumatoria estándar de armónicos impares
2. **Onda Diente de Sierra:** Progresión lineal armónica.
3. **Onda Triangular:** Atenuación cuadrática con signos alternados.
4. **Tren de Pulsos:** Modulación simétrica de alta frecuencia.
5. **Sierra Asimétrica:** Modulación progresiva impar.
6. **Pulso Cuadrático:** Sumatoria compleja con decaimiento cuadrático exponencial.

---

## 🖼️ Evidencia de Ejecución

**📈 Modo Fórmula y Paramétrico**

Muestra de superficies complejas generadas mediante el sistema:

- **Superficies:** Botella de Klein, Helicoides, Sillas de montar.
- **Render:** Totalmente interactivo en el navegador.

**🎵 Modo Funciones Sonoras**

Generación de espectros tridimensionales y curvas analíticas 2D acopladas a ecuaciones matemáticas dinámicas en LaTeX.

### _[!TIP]_

Si el modelo 3D aparece invertido o con inconsistencias en las caras, revisa la función _sort_contour_points_ en _extrusion.py_ o el orden de puntos en _contours.py_.

---

## 🤝 Flujo de Contribución

Seguimos una metodología estricta basada en tickets para mantener la trazabilidad:

- **Localizar Ticket:** Identificar el ID (ej. VEK-009).
- **Crear Rama:**

  ```bash
  git checkout -b feat/VEK-009-descripcion-corta

  ```

- **Commit Standard:**
  ```bash
  "feat: descripción del cambio (VEK-009)"
  ```

---

## 📂 Estructura del Proyecto

- **/modules/geometry:** Contiene las capas core de modelado matemático (primitives.py, parametric.py, extrusion.py y fourier.py).
- **/modules/vision:** Procesamiento digital de imágenes y extracción de contornos (image_processing.py, contours.py).
- **/ui:** Vistas modulares de la interfaz de usuario en Streamlit (tab\_\*.py).
- **app.py:** Punto de entrada principal del layout general de la aplicación.
- **requirements.txt:** Archivo de especificación de dependencias del entorno.
