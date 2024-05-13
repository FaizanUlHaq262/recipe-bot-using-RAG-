from flask import Flask, render_template, request, jsonify, send_file
import io
from transformers import pipeline
import rag_chatbot

app = Flask(__name__)
pipe = pipeline("text-to-speech", model="suno/bark-small")

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/tts", methods=["POST"])
def text_to_speech():
    text = request.form['text']
    output = pipe(text)
    audio_bytes = output["audio"]
    return send_file(
        io.BytesIO(audio_bytes),
        mimetype="audio/wav",
        as_attachment=True,
        download_name="speech.wav"
    )

@app.route("/ask", methods=['POST'])
def ask():
    user_input = request.form['message']
    if user_input.startswith('/recipe '):
        response = rag_chatbot.makeOllamaQuery(
            vectorDB=rag_chatbot.phidbb3,
            model='phi3',
            numResults=5,
            promptQuery=rag_chatbot.promptRecipeQuery,
            ragPromptTemplate=rag_chatbot.template,
            userInp=user_input
        )
    elif user_input.startswith('/recommend '):
        response = rag_chatbot.makeOllamaQuery(
            vectorDB=rag_chatbot.phidbb3,
            model='phi3',
            numResults=5,
            promptQuery=rag_chatbot.promptRecommendQuery,
            ragPromptTemplate=rag_chatbot.template,
            userInp=user_input
        )
    elif user_input.startswith('/how '):
        response = rag_chatbot.makeOllamaQuery(
            vectorDB=rag_chatbot.phidbb3,
            model='phi3',
            numResults=5,
            promptQuery=rag_chatbot.promptHowQuery,
            ragPromptTemplate=rag_chatbot.template,
            userInp=user_input
        )
    else:
        response = "Use correct syntax."
    return jsonify({'message': response})

if __name__ == "__main__":
    app.run(debug=True)
