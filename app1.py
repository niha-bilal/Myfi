import gradio as gr
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import os
import requests
from langchain.llms import Ollama
from langdetect import detect

# === LangChain Tracing Config ===
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# === Load fine-tuned BERT model ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "intent_model")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
labels = np.load("label_classes.npy", allow_pickle=True)

# === Load prompt examples from Google Sheet ===
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTf2Tm2H0Yvqs7-g5n_ysK0QYd0mVhhPKArdd7s-Z06mKd7UV4fjOJjbgUVODhqmXpk4_-OQHdyEnjn/pub?output=csv"
df_examples = pd.read_csv(CSV_URL).dropna()

# === LLaMA 3.2 via LangChain + Ollama ===
llm = Ollama(model="llama3.2")

# === Helper: Build few-shot prompt ===
def build_phi2_prompt(user_text):
    prompt = "Intent Classification:\n"
    for _, row in df_examples.iterrows():
        prompt += f"Text: {row['text']}\nIntent: {row['intent']}\n"
    prompt += f"Text: {user_text}\nIntent:"
    return prompt

# === Helper: Detect Language ===
def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "en"

# === Classifier: BERT ===
def classify_with_bert(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return labels[pred]

# === Classifier: Phi-2 / LLaMA 3.2 ===
def classify_with_phi2(text):
    prompt = build_phi2_prompt(text)
    try:
        response = llm.invoke(prompt)
        return response.strip().split("\n")[0]
    except Exception:
        return "UNKNOWN"

# === LLM-Based Reply Generation ===
def generate_text_response(text, intent, chat_history):
    conversation = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    lang = detect_language(text)

    if lang == "hi":
        sys_prompt = "आप SOF मैट्रेस ग्राहक सहायता सहायक हैं। अब तक की बातचीत:\n"
        customer_label = "ग्राहक"
        assistant_label = "सहायक"
    else:
        sys_prompt = "You are a helpful assistant for SOF mattress customers. Here is the chat so far:\n"
        customer_label = "Customer"
        assistant_label = "Assistant"

    full_prompt = f"{sys_prompt}{conversation}\n{customer_label}: {text}\n{assistant_label}:"
    try:
        return llm.invoke(full_prompt).strip()
    except Exception:
        return "Sorry, something went wrong while generating a response."

# === Chatbot Logic ===
def chatbot(user_msg, history):
    if history is None:
        history = []

    bert_intent = classify_with_bert(user_msg)
    phi_intent = classify_with_bert(user_msg)
    reply = generate_text_response(user_msg, bert_intent, history)

    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": reply})

    return history, bert_intent, phi_intent, history

# === Gradio UI ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 SOF Intent Detection + Response Generator\nUsing BERT & LLaMA3.2 via Ollama")

    chat_ui = gr.Chatbot(label="💬 Chat", type="messages", show_copy_button=True)
    user_input = gr.Textbox(label="Type your message", placeholder="Ask about EMI, COD, mattress features...", lines=2)
    intent_bert = gr.Textbox(label="🎯 Intent (BERT)", interactive=False)
    intent_phi = gr.Textbox(label="🔮 Intent (LLaMA 3.2)", interactive=False)
    state = gr.State([])

    with gr.Row():
        user_input.submit(fn=chatbot, inputs=[user_input, state], outputs=[chat_ui, intent_bert, intent_phi, state])

demo.launch()
