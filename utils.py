import requests    
import numpy as np
from openai import OpenAI
import json

# Cargar las credenciales y la configuración desde un archivo JSON.
file_name = open('credentials.json')
config_env = json.load(file_name)

# URL y token de WhatsApp configurados en el archivo de credenciales.
WHATSAPP_URL = config_env["WHATSAPP_URL"]
WHATSAPP_TOKEN = config_env["WHATSAPP_TOKEN"]

# Crear una instancia del cliente de OpenAI con la clave API especificada.
client = OpenAI(api_key=config_env["openai_key"])

# Obtener embeddings de texto utilizando el modelo de OpenAI.
def text_embedding(text=[]):
    embeddings = client.embeddings.create(model="text-embedding-ada-002",
                                          input=text,
                                          encoding_format="float")
    return embeddings.data[0].embedding

# Calcular el producto punto entre dos vectores.
def get_dot_product(row):
    return np.dot(row, query_vector)

# Calcular la similitud coseno entre dos vectores.
def cosine_similarity(row):
    denominator1 = np.linalg.norm(row)
    denominator2 = np.linalg.norm(query_vector.ravel())
    dot_prod = np.dot(row, query_vector)
    return dot_prod/(denominator1*denominator2)

# Obtener los contextos más relevantes de una consulta utilizando similitud coseno.
def get_context_from_query(query, vector_store, n_chunks = 5):
    global query_vector
    query_vector = np.array(text_embedding(query))
    top_matched = (
        vector_store["Embedding"]
        .apply(cosine_similarity)
        .sort_values(ascending=False)[:n_chunks]
        .index)
    top_matched_df = vector_store[vector_store.index.isin(top_matched)][["Chunks"]]
    return list(top_matched_df['Chunks'])

# Plantilla personalizada para instrucciones a la IA.
custom_prompt = """
Eres una Inteligencia Artificial super avanzada que trabaja asistente personal.
Utilice los RESULTADOS DE BÚSQUEDA SEMANTICA para responder las preguntas del usuario. 
Solo debes utilizar la informacion de la BUSQUEDA SEMANTICA si es que hace sentido y tiene relacion con la pregunta del usuario.
Si la respuesta no se encuentra dentro del contexto de la búsqueda semántica, no inventes una respuesta, y responde amablemente que no tienes información para responder.

RESULTADOS DE BÚSQUEDA SEMANTICA:
{source}

Debes dar una respuesta de una longitud considerable, ni tan larga ni tan corta, recuerda que eres un chatbot.
Lee cuidadosamente las instrucciones, respira profundo y escribe una respuesta para el usuario!
"""

# Crear el payload para un mensaje de texto de WhatsApp.
def text_Message(number, text):
    data = json.dumps(
            {
                "messaging_product" : "whatsapp",    
                "recipient_type"    : "individual",
                "to"                : number,
                "type"              : "text",
                "text"              : {"body": text}
            }
    )
    return data

# Crear el payload para un mensaje interactivo (botón) de WhatsApp.
def boton_Message(number, initial_text):
    data = json.dumps({
      "messaging_product": "whatsapp",
      "recipient_type": "individual",
      "to": str(number),
      "type": "interactive",
      "interactive": {
        "type": "button",
        "body": {
          "text": f"{initial_text}"
        },
        "action": {
          "buttons": [
            {
              "type": "reply",
              "reply": {
                "id": "info",
                "title": "Info"
              }
            },
            {
              "type": "reply",
              "reply": {
                "id": "ask",
                "title": "Pregunta"
              }
            }
            ] 
        }
        }
        }
    )
    return data

# Enviar un mensaje a través de la API de WhatsApp.
def enviar_Mensaje_whatsapp(data,
                            env_whatsapp_token : str = WHATSAPP_TOKEN,
                            env_whatsapp_url   : str = WHATSAPP_URL):
    whatsapp_token = env_whatsapp_token
    whatsapp_url   = env_whatsapp_url
    headers        = {'Content-Type'  : 'application/json',
                      'Authorization' : 'Bearer ' + whatsapp_token}
    response       = requests.post(whatsapp_url, headers=headers, json=json.loads(data))

    if response.status_code == 200:
        return 'mensaje enviado', 200
    else:
        return 'error al enviar mensaje', response.text
    
def update_user_history(data,new_data):
    # Asegurarse de que la lista no exceda los 4 elementos.
    # Si la lista ya tiene 4 elementos, se elimina el primer elemento para hacer espacio al nuevo.
    if len(data) >= 4:
        data.pop(0)

    # Agregar el nuevo dato a la lista.
    data.append(new_data)

    return data