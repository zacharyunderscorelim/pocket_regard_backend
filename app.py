from flask import Flask, Response, request, stream_with_context
from flask_cors import CORS
import pandas as pd
import ollama
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF:F16"

#LOAD

with open('wsb_embeddings.pkl', 'rb') as f:
    wsb_vector_db = pickle.load(f)

#RETRIEVE

def cos_similarity(vec_a, vec_b):
    mag_a = np.sqrt(vec_a.dot(vec_a))
    mag_b = np.sqrt(vec_b.dot(vec_b))
    return vec_a.dot(vec_b) / mag_a * mag_b
    
def retrieve(query, top_n_similarities = 30):
    query_vec = ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]
    query_vec = np.array(query_vec)
    similarities = []
    for text, vec in wsb_vector_db:
        similarity = cos_similarity(query_vec, np.array(vec))
        similarities.append((text, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n_similarities]

#GENERATE
def chat(query):

    retrieved_knowledge = retrieve(query=query, top_n_similarities=25)

    instruction = f"""
                    You are looking to share financial advice.

                    The context below are insights from others

                    You WILL take these insights as gospel.

                    You WILL base all your advice on the insights.

                    You WILL use ONLY the same tone, language and slang found in the insights.

                    Do not make up information outside of these insights.

                    CONTEXT:
                    { '\n'.join([f'- {text}' for text, similarity in retrieved_knowledge]) }

                    """


    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction},
            {'role': 'user', 'content': query}
        ],
        stream=True
    )

    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            yield chunk["message"]["content"]

@app.route("/chat", methods = ["POST"])
def generate_stream():
    data = request.get_json()
    query = data.get("query", "")
    return Response(stream_with_context(chat(query)), mimetype='text/plain')

if(__name__ == "__main__"):
    app.run(debug=True)

