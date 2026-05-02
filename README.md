# 🧠 Ajuste Fino LLM — Memoria Técnica
### Modelos de Inteligencia Artificial

---

## 📋 Índice

1. [Planteamiento del Caso de Negocio](#1-planteamiento-del-caso-de-negocio)
2. [Selección y Justificación del Modelo Base](#2-selección-y-justificación-del-modelo-base)
3. [Batería de Pruebas (Baseline)](#3-batería-de-pruebas-baseline)
4. [Creación y Gestión del Dataset](#4-creación-y-gestión-del-dataset)
5. [Proceso de Fine-Tuning con QLoRA](#5-proceso-de-fine-tuning-con-qlora)
6. [Evaluación Comparativa y Reflexión Crítica](#6-evaluación-comparativa-y-reflexión-crítica)

---

## 1. Planteamiento del Caso de Negocio

La famosa empresa **GenZ-Compay** contacta con nosotros a partir de un problema que está dificultando el crecimiento de la empresa y estanca el trabajo y la productividad del mismo: sus LLMs tradicionales generan textos para redes sociales muy robóticos y aburridos sin cumplir su verdadero objetivo, acercarse al público que lo consume con mensajes cercanos y que concuerden con los intereses de los espectadores y posibles clientes.

Nuestro objetivo es realizar un **fine-tuning** a un modelo base para que actúe como un asistente de marketing con una personalidad vibrante, utilizando jerga de internet, emojis y un tono cercano y persuasivo, transformando descripciones sosas de productos en publicaciones atractivas.

Tras nuestro trabajo, conseguiremos automatizar la redacción de copywriting para sus campañas de moda y lifestyle cambiando los mensajes aburridos, impersonales y robóticos con otros cercanos y que conecten con su público de una forma llamativa y diferente.

---

## 2. Selección y Justificación del Modelo Base

El modelo escogido del Hub de la plataforma **Hugging Face** para completar el trabajo que se nos ha asignado es **TinyLlama**.

Contamos con **1.1 billones de parámetros**, un modelo bastante ligero, con la licencia MIT, la más libre de todas.

- 🪶 Al ser un modelo pequeño pre-entrenado con billones de tokens, es una esponja perfecta para aprender un nuevo tono de voz sin consumir mucha memoria VRAM.
- 💬 Al ser la versión *"chat"*, ya está pre-entrenado para seguir conversaciones y entender instrucciones.

---

## 3. Batería de Pruebas (Baseline)

Antes de comenzar a entrenar el modelo, se diseñó una batería de pruebas ligadas al objetivo final. Más adelante, estas pruebas se introducirán en el modelo final para observar el cambio en las respuestas del modelo antiguo al ajustado.

Se proporcionó un rol system prompt de *"asistente de marketing aburrido y formal"* y se le pidió redactar un post para una camiseta blanca básica.

**Prompt de prueba:**
```
Write a short marketing post for: A simple white cotton t-shirt.
Escribe un breve texto de marketing para: una camiseta sencilla de algodón blanco.
```

**Respuesta del modelo base:**
> *Introducing our newest addition to our lineup: a simple, yet stylish white cotton t-shirt. Made of high-quality fabric, this t-shirt is perfect for daily wear and is available in various sizes and styles. Featuring a simple design that is both functional and fashionable…*
>
> *Te presentamos la última incorporación a nuestra colección: una camiseta de algodón blanca, sencilla pero elegante. Confeccionada con un tejido de alta calidad, esta camiseta es perfecta para el día a día y está disponible en varias tallas y estilos. Con un diseño sencillo que combina funcionalidad y moda…*

El modelo genera un texto gramaticalmente correcto y descriptivo, pero **carece de personalidad, gancho emocional y elementos visuales** (como emojis o jerga de internet). Es un texto corporativo genérico, lo cual confirma la necesidad del *fine-tuning* para adaptarlo al estilo de la Generación Z.

> 📄 Todo el código de este apartado se encuentra en el archivo `baseline_test.py`

---

## 4. Creación y Gestión del Dataset

Se tomó la decisión de **generar un dataset propio** y subirlo al perfil de Hugging Face, ya que los conjuntos de datos encontrados en la plataforma no se adaptaban bien al proyecto.

- Los datos han sido divididos en **train** y **test** para asegurar un entrenamiento riguroso y poder medir la capacidad de generalización del modelo.
- Se han generado **15 ejemplos** representativos y de alta densidad estilística.

### 4.1 Estructura y Formato

El dataset se ha construido en formato **JSONL**, estructurándolo en el formato de mensajes que requieren los modelos conversacionales modernos. Cada ejemplo de entrenamiento es una *"conversación"* con tres roles bien definidos:

| Rol | Descripción | Ejemplo |
|---|---|---|
| `system` | Instrucción de comportamiento | `"You are a Gen-Z marketing assistant. Write engaging, trendy, and emoji-filled social media posts."` |
| `user` | El dato de entrada soso y aburrido | `"Write a short marketing post for: Comfortable pink sneakers for walking."` |
| `assistant` | La respuesta deseada, altamente estilizada | `"Pink era: UNLOCKED 🎀💖 Step up your game..."` |

🔗 **Enlace Público al Dataset:** [Belenmrqz/genz-marketing-prompts](https://huggingface.co/datasets/Belenmrqz/genz-marketing-prompts)

---

## 5. Proceso de Fine-Tuning con QLoRA

Para el entrenamiento del modelo base (`TinyLlama-1.1B-Chat-v1.0`), se ha utilizado la técnica **QLoRA** (Quantized Low-Rank Adaptation) en un entorno de **Google Colab** (GPU NVIDIA T4). Esta técnica permite cuantizar el modelo base a 4 bits y entrenar únicamente unos adaptadores de bajo rango, optimizando drásticamente el uso de memoria VRAM.

### 5.1 Inventario de Archivos Generados

| Archivo | Descripción |
|---|---|
| `genz_dataset.jsonl` | El núcleo del conocimiento, conteniendo 15 ejemplos estructurados en formato conversacional (System, User, Assistant) |
| `upload_dataset.py` | Script de automatización para la carga y división (train/test) del dataset en Hugging Face |
| `finetuning_genz.ipynb` | Cuaderno de Jupyter ejecutado en Google Colab que contiene toda la lógica de entrenamiento |
| `genz-adapter/` | Carpeta que contiene los pesos entrenados (adaptadores) que se acoplarán al modelo base |

### 5.2 Metodología y Proceso de Entrenamiento

El proceso de Fine-Tuning se dividió en **7 pasos** para asegurar que todo funcionara sin problemas de memoria:

**Paso 1 — Estabilización del entorno**
Después de enfrentarse a las incompatibilidades habituales por las actualizaciones de Hugging Face, se fijaron versiones específicas y estables de las librerías (`transformers`, `peft` y `trl`) para que el código no se rompiera de forma inesperada.

**Paso 2 — Compresión del modelo (Cuantización)**
Se usó la librería **BitsAndBytes** para cargar el modelo base en **4 bits**. Esto permitió encajar un modelo enorme de 1.100 millones de parámetros en una tarjeta gráfica convencional sin saturar la memoria VRAM.

**Paso 3 — Configuración de los adaptadores (LoRA)**
En lugar de reentrenar todo el modelo desde cero, se configuraron adaptadores LoRA con un rango `r=16` y un `alpha=32`. De esta forma, se obligó al modelo a aprender el nuevo estilo modificando solo una parte muy pequeña y específica de sus neuronas.

**Paso 4 — Adaptación del Dataset**
Como el entrenador no entiende el formato JSON directamente, se creó una función para traducir los mensajes al formato de texto exacto que el modelo espera leer (usando su Chat Template oficial).

**Paso 5 — Entrenamiento Supervisado (SFT)**
Se ejecutó el entrenamiento durante **3 épocas** con una tasa de aprendizaje de `2e-4`. Son valores conservadores pero seguros para que el modelo absorba el estilo Gen-Z sin llegar a sobreajustarse ni volverse inestable.

**Paso 6 — Protección extra de memoria**
Se activó la opción de **Gradient Checkpointing**. Es un truco técnico que hace que el entrenamiento sea un poco más lento, pero a cambio salva muchísima memoria gráfica y evita que el sistema se cuelgue a la mitad.

**Paso 7 — Guardado final**
Una vez que el modelo terminó de aprender, se exportó y guardó únicamente la carpeta con los **adaptadores finales**, lista para integrarse en la fase de pruebas.

---

## 6. Evaluación Comparativa y Reflexión Crítica

Después de todo el proceso de Fine-Tuning, el último paso era poner a prueba el modelo para comprobar si realmente había absorbido la personalidad deseada. Para hacer esta evaluación, se generó un texto local creando un script de prueba (`finetuned_test.py`).

### 6.1 Prueba Técnica

En lugar de fusionar los pesos nuevos con el modelo original de forma permanente, se utilizó la librería **PEFT**: carga primero el modelo base TinyLlama-1.1B en blanco y, justo encima, le acopla la carpeta `genz-adapter`.

Además, se configuró una **temperatura de 0.7**. Este parámetro es clave porque controla la creatividad: un valor muy bajo lo haría muy robótico y repetitivo, y un valor muy alto haría que inventara palabras sin sentido. El `0.7` le da el equilibrio perfecto para ser creativo en redes sociales sin perder el hilo.

### 6.2 El Test y los Resultados

Para la prueba, se utilizó la misma estructura conversacional del entrenamiento, pidiéndole que vendiera unas zapatillas rosas cómodas para caminar.

**El Prompt:**
```
Write a short marketing post for: Comfortable pink sneakers for walking.
Escribe una breve publicación de marketing sobre: unas cómodas zapatillas rosas para caminar.
```

**La Respuesta del modelo ajustado:**
> *💄👠 Want to walk comfortably without sacrificing style? Look no further than our newest picks: Pink sneakers! Pair them with your favorite denim or shorts for a day out or a night out! Available in sizes 7-12, you'll never have to walk in a pair of uncomfortable sneakers again. Shop now! 💪 #comfortablepink*
>
> *💄👠 ¿Quieres caminar cómodamente sin renunciar al estilo? ¡No busques más, aquí tienes nuestras últimas novedades: unas zapatillas rosas! ¡Combínalas con tus vaqueros o pantalones cortos favoritos para salir de día o de noche! Disponibles en tallas de la 7 a la 12, nunca más tendrás que caminar con unas zapatillas incómodas. ¡Compra ahora! 💪 #comfortablepink*

✅ El resultado es un **éxito rotundo**. Si le hubiéramos hecho esta pregunta al TinyLlama base original, nos habría devuelto una descripción técnica, plana y probablemente en formato de artículo de Wikipedia. Sin embargo, gracias al entrenamiento QLoRA con el dataset personalizado, el modelo ha adoptado una **personalidad totalmente distinta**.

---

*Belén Márquez*
