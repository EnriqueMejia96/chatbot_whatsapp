from flask import Flask, request, render_template
from utils import *
import json
import pandas as pd

# Cargar las credenciales y configuraciones iniciales desde un archivo JSON.
file                = open('credentials.json')
config_env          = json.load(file)

# Clave de autenticación para WhatsApp.
WHATSAPP_KEY = config_env["WHATSAPP_KEY"]

from openai import OpenAI
# Cargar un DataFrame de pandas previamente almacenado en un archivo pickle.
# Este DataFrame contiene embeddings vectoriales y otros datos relevantes.
df_vector_store = pd.read_pickle('df_vector_store.pkl')

# Inicializar la aplicación Flask.
app = Flask(__name__)

# Definir la ruta de inicio que simplemente devuelve una confirmación de que todo está bien.
@app.route('/')
def home():
        return 'All is well...'

# Ruta para verificar el token enviado por WhatsApp en las solicitudes GET.
@app.route('/webhook', methods=['GET'])
def verificar_token():
    try:
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')

        if token == WHATSAPP_KEY and challenge != None:
            return challenge
        else:
            return 'token incorrecto', 403
    except Exception as e:
        return e,403

# Ruta para recibir mensajes vía POST en el webhook.
@app.route('/webhook', methods=['POST'])
def recibir_mensajes():
    try:
        # Obtener el cuerpo del mensaje en formato JSON.
        body = request.get_json()
        entry = body['entry'][0]
        changes = entry['changes'][0]
        value = changes['value']
        message = value['messages'][0]
        message_type = message['type']
        number = message['from']

        # Procesar mensajes interactivos (por ejemplo, respuestas a botones).
        if message_type=='interactive':
            text = message['interactive']['button_reply']['title']

            if text == 'Info':
                response_ = 'Información: Este es un chatbot de prueba!'
            elif text == 'Pregunta':
                response_ = 'Por supuesto, ¿Cuál es tu consulta?'

            data = text_Message(number = number,
                                text   = response_)
            enviar_Mensaje_whatsapp(data)

        # Procesar mensajes de texto normales.
        elif message_type == 'text':
            text = message['text']['body']

            if text == 'Hola':
                response_ = '¡Hola! Soy tu ChatBot personal, estoy aquí para solucionar temas sobre LLMs'
                data = boton_Message(number = number,
                                      initial_text = response_)
                enviar_Mensaje_whatsapp(data) 

            else:
                # Cargar el historial del usuario desde un archivo JSON.
                with open('user_history.json', 'r') as file:
                    user_history = json.load(file)

                # Obtener contextos relevantes del almacenamiento vectorial basado en la consulta.
                Context_List = get_context_from_query(query        = text,
                                                      vector_store = df_vector_store,
                                                      n_chunks     = 5)
                # Crear una instancia del cliente de OpenAI con la clave API.
                client = OpenAI(api_key=config_env["openai_key"])
                # Generar una respuesta utilizando el modelo y la configuración seleccionados por el usuario.
                completion = client.chat.completions.create(
                model="gpt-4",
                temperature = 0.0,
                messages=[{"role": "system", "content": f"{custom_prompt.format(source = str(Context_List))}"}] + 
                            user_history + 
                            [{"role": "user", "content": text}])
                response_ = completion.choices[0].message.content

                # Enviar mensaje a whatsapp
                data = text_Message(number = number,
                                    text   = response_)
                enviar_Mensaje_whatsapp(data)

                # Actualizar el historial del usuario con la nueva consulta y respuesta.
                user_history = update_user_history(data = user_history,
                                                   new_data = {"role": "user", "content": text})
                user_history = update_user_history(data = user_history,
                                                   new_data = {"role": "assistant", "content": response_})
                with open('user_history.json', 'w') as file:
                    json.dump(user_history, file, indent=4)

        # Procesar otros tipos de mensajes como audio, imagen o video.
        elif message_type == 'audio' or message_type == 'image' or message_type == 'video':
            response_ = f'Disculpa no puedo recibir {message_type}s, dime si tienes alguna consulta adicional.'
            data = text_Message(number = number,
                                text   = response_)
            enviar_Mensaje_whatsapp(data)
        return 'enviado'

    except Exception as e:
        # Registrar cualquier excepción en un archivo de logs.
        # with open('logs.txt', "w") as file:
        #     file.write(str(e))
        return 'no enviado'

# Ejecutar la aplicación Flask en el host y puerto especificados.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)