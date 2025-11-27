import gradio as gr
import pickle
import plotly.express as px

# Carga tu modelo y vectorizador
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Frases predefinidas
sample_reviews = [
    "The location is excellent, only two minutes from the subway.",
    "The bed was really comfortable and I slept very well.",
    "The hotel is simple, but it delivers exactly what it promises.",
    "Everything was so incredible that I don't understand how anyone could ever complain.",
    "There is no better hotel anywhere in the city, guaranteed.",
    "This hotel is so perfect it should have ten stars.",
]

def predict_text(selected_text, written_text):
    # Elegir texto: prioridad al texto escrito
    text = written_text.strip() if written_text.strip() else selected_text

    # Preprocesar texto
    X = tfidf.transform([text])
    pred = model.predict(X)[0]
    proba = model._predict_proba_lr(X)[0]

    fig = px.bar(
        x=["Real", "Manipulada"],
        y=proba,
        title="Probabilidad del Modelo",
        labels={"x": "Clase", "y": "Probabilidad"},
    )

    return f"**Texto analizado:** {text}\n\n**Predicción:** **{pred}**", fig


# Interfaz Gradio
app = gr.Interface(
    fn=predict_text,
    inputs=[
        gr.Dropdown(sample_reviews, label="Selecciona una reseña de prueba"),
        gr.Textbox(lines=5, label="O escribe tu propia reseña…"),
    ],
    outputs=[
        gr.Markdown(label="Resultado"),
        gr.Plot(label="Probabilidades"),
    ],
    title="Clasificador de Opiniones Engañosas (TF-IDF + ML)",
    description="Modelo entrenado en deceptive-opinion-spam-corpus. Puedes seleccionar una reseña o escribir la tuya.",
)

app.launch(debug=True)
