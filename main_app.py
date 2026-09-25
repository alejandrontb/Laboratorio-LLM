import os
import streamlit as st
import numpy as np
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pytesseract

# Configuración de la página
st.set_page_config(
    page_title="Plataforma NLP & Groq Studio",
    page_icon="🤖",
    layout="wide"
)

# --- BARRA LATERAL: CONFIGURACIÓN DE API KEY ---
st.sidebar.header("🔑 Configuración")
api_key_input = st.sidebar.text_input("Ingresa tu API Key de Groq", type="password")

if api_key_input:
    os.environ["GROQ_API_KEY"] = api_key_input
    st.sidebar.success("¡API Key configurada!")
else:
    st.sidebar.warning("Por favor, ingresa tu API Key para usar los modelos de Groq.")

st.sidebar.markdown("---")
menu = st.sidebar.options = st.sidebar.radio(
    "Navegación",
    ["1. Tokenización y Colores", "2. Bag of Words", "3. Similitud de Coseno", "4. Groq Playground & Modelos", "5. OCR + Prompt Generator"]
)

# --- MÓDULO 1: TOKENIZACIÓN ---
if menu == "1. Tokenización y Colores":
    st.header("🔠 Esquemas de Tokenización y Visualización")
    st.write("Analiza cómo se dividen las palabras o caracteres y visualiza los tokens con colores dinámicos.")

    text_input = st.text_area("Ingresa un texto para tokenizar:", "La inteligencia artificial generativa transforma el desarrollo de software.")
    
    # Selector de esquema simple basado en espacios/caracteres o simulación de tokens
    schema = st.selectbox("Selecciona el esquema de tokenización", ["Por Palabras (Whitespace)", "Por Caracteres", "Simulación Subword (N-grams)"])

    if st.button("Tokenizar"):
        if schema == "Por Palabras (Whitespace)":
            tokens = text_input.split()
        elif schema == "Por Caracteres":
            tokens = list(text_input)
        else:
            # Simulación simple de subwords dividiendo cada 4 caracteres o palabras
            tokens = [word[i:i+4] for word in text_input.split() for i in range(0, len(word), 4)]

        st.subheader("Resultados:")
        
        # Mostrar Token IDs simulados
        token_ids = [hash(t) % 10000 for t in tokens]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Total de tokens:** {len(tokens)}")
            st.write("**Tokens IDs (Hash simulado):**", token_ids)

        # Mostrar tokens con colores
        st.markdown("### Visualización de Tokens con Colores")
        colors = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3", "#33FFF0"]
        
        colored_html = ""
        for i, token in enumerate(tokens):
            bg_color = colors[i % len(colors)]
            colored_html += f'<span style="background-color: {bg_color}; padding: 4px 8px; margin: 2px; border-radius: 4px; display: inline-block; color: #000; font-weight: bold;">{token} <sub style="font-size: 9px;">ID:{token_ids[i]}</sub></span> '
        
        st.markdown(colored_html, unsafe_allow_html=True)

# --- MÓDULO 2: BAG OF WORDS ---
elif menu == "2. Bag of Words":
    st.header("📊 Bag of Words (Bolsa de Palabras)")
    st.write("Genera la matriz de conteo de palabras para un conjunto de frases.")

    corpus_input = st.text_area(
        "Ingresa frases (una por línea):",
        "Me gusta aprender inteligencia artificial\nGroq ofrece una inferencia muy rápida\nLa inteligencia artificial es el futuro"
    )

    if st.button("Generar Bag of Words"):
        documents = [doc.strip() for doc in corpus_input.split("\n") if doc.strip()]
        if len(documents) > 0:
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(documents)
            
            st.subheader("Vocabulario detectado:")
            st.write(vectorizer.get_feature_names_out())

            st.subheader("Matriz BoW:")
            st.write(X.toarray())
        else:
            st.error("Por favor, ingresa al menos una frase válida.")

# --- MÓDULO 3: SIMILITUD DE COSENO ---
elif menu == "3. Similitud de Coseno":
    st.header("📐 Similitud de Coseno entre Frases")
    st.write("Calcula la cercanía semántica/léxica entre dos frases.")

    frase1 = st.text_input("Frase 1:", "El clima de hoy está soleado y agradable.")
    frase2 = st.text_input("Frase 2:", "Hoy hace un día maravilloso con mucho sol.")

    if st.button("Calcular Similitud"):
        vectorizer = CountVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([frase1, frase2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            
            score = similarity[0][0]
            st.metric(label="Similitud de Coseno", value=f"{score:.4f}")
            
            if score > 0.7:
                st.success("Las frases son altamente similares.")
            elif score > 0.3:
                st.info("Las frases tienen similitud moderada.")
            else:
                st.warning("Las frases son bastante diferentes.")
        except Exception as e:
            st.error(f"Error al calcular: {e}")

# --- MÓDULO 4: GROQ PLAYGROUND & MODELOS ---
elif menu == "4. Groq Playground & Modelos":
    st.header("🚀 Groq Playground & Catálogo de Modelos")
    st.write("Interactúa con los modelos oficiales disponibles en la API de Groq ajustando parámetros avanzados.")

    # Catálogo de modelos "No llama" (No usar GPT)
    groq_models = {
        "Llama 3 8B (Versátil)": "llama3-8b-8192",
        "Llama 3 70B (Avanzado)": "llama3-70b-8192",
        "Mixtral 8x7B (Balanceado)": "mixtral-8x7b-32768",
        "Gemma 7B (Ligero)": "gemma-7b-it"
    }

    selected_model_name = st.selectbox("Selecciona un Modelo Groq:", list(groq_models.keys()))
    model_id = groq_models[selected_model_name]

    # Parámetros avanzados
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperatura (Creatividad)", 0.0, 2.0, 0.7, 0.1)
    with col2:
        max_tokens = st.slider("Max Tokens (Longitud)", 100, 4000, 1024, 100)

    prompt = st.text_area("Escribe tu Prompt:", "Explica brevemente qué es la computación cuántica.")

    if st.button("Enviar a Groq"):
        if not api_key_input:
            st.error("Por favor, configura tu API Key en la barra lateral.")
        else:
            try:
                client = Groq(api_key=api_key_input)
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                st.subheader("Respuesta del Modelo:")
                st.write(chat_completion.choices[0].message.content)
            except Exception as e:
                st.error(f"Ocurrió un error al conectar con Groq: {e}")

# --- MÓDULO 5: OCR + PROMPT EXTENSION ---
elif menu == "5. OCR + Prompt Generator":
    st.header("📷 OCR + Ampliación de Prompt con Groq")
    st.write("Sube una imagen con texto. El sistema extraerá el contenido mediante OCR y lo usará como base para generar una respuesta enriquecida con Groq.")

    uploaded_file = st.file_uploader("Sube una imagen (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)

        if st.button("Extraer texto con OCR"):
            try:
                extracted_text = pytesseract.image_to_string(image)
                st.subheader("Texto Extraído:")
                st.text_area("Resultado OCR:", extracted_text, height=150)
                st.session_state["extracted_ocr_text"] = extracted_text
            except Exception as e:
                st.error(f"Error al procesar el OCR (asegúrate de tener Tesseract instalado): {e}")

    # Si ya hay texto extraído por OCR, permitir usarlo como prompt
    if "extracted_ocr_text" in st.session_state and st.session_state["extracted_ocr_text"]:
        st.markdown("---")
        st.subheader("Ampliar respuesta usando el texto del OCR")
        
        extra_instruction = st.text_input("Instrucción adicional para complementar el texto OCR:", "Resume y analiza los puntos clave de este texto.")
        
        if st.button("Generar respuesta ampliada con Groq"):
            if not api_key_input:
                st.error("Por favor, ingresa tu API Key en la barra lateral.")
            else:
                full_prompt = f"{extra_instruction}\n\nTexto base de la imagen:\n{st.session_state['extracted_ocr_text']}"
                try:
                    client = Groq(api_key=api_key_input)
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": full_prompt}],
                        model="llama3-8b-8192",
                        temperature=0.5
                    )
                    st.subheader("Respuesta Ampliada:")
                    st.write(chat_completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error al conectar con Groq: {e}")
