import gradio as gr
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import os
from langchain.llms import Ollama
from langdetect import detect
import pandas as pd
import requests
import json


# === Load prompt data from Google Sheet ===
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTf2Tm2H0Yvqs7-g5n_ysK0QYd0mVhhPKArdd7s-Z06mKd7UV4fjOJjbgUVODhqmXpk4_-OQHdyEnjn/pub?output=csv"
df_examples = pd.read_csv(CSV_URL).dropna()

# Build few-shot prompt from dataset
def build_phi2_prompt(user_text):
    prompt = "Intent Classification:\n"
    for _, row in df_examples.iterrows():
        prompt += f"Text: {row['text']}\nIntent: {row['intent']}\n"
    prompt += f"Text: {user_text}\nIntent:"
    return prompt

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

# === Keyword override rules ===
HIGH_CONFIDENCE_KEYWORDS = {
    "warranty": "WARRANTY",
    "trial": "100_NIGHT_TRIAL_OFFER",
    "emi": "EMI",
    "cod": "COD"
}

def classify_with_bert(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze()

    threshold = 0.6
    predicted_indices = (probs > threshold).nonzero(as_tuple=True)[0].tolist()

    if not predicted_indices:
        predicted_indices = torch.topk(probs, k=1).indices.tolist()

    predicted_intents = [labels[i] for i in predicted_indices]

    # High-confidence override
    lower_text = text.lower()
    for keyword, override_intent in HIGH_CONFIDENCE_KEYWORDS.items():
        if keyword in lower_text and override_intent not in predicted_intents:
            predicted_intents.insert(0, override_intent)

    # Debug print
    print("\n🔍 Intent Probabilities:")
    for i, p in enumerate(probs.tolist()):
        print(f"{labels[i]}: {p:.3f}")

    return ", ".join(predicted_intents)


def keyword_override(text, predicted_intents):
    lower_text = text.lower()
    for keyword, intent in HIGH_CONFIDENCE_KEYWORDS.items():
        if keyword in lower_text and intent not in predicted_intents:
            predicted_intents.insert(0, intent)
    return predicted_intents

    return ", ".join(keyword_override(text, [labels[i] for i in predicted_indices]))


# === LLaMA 3.2 via LangChain + Ollama ===
llm = Ollama(model="llama3.2")

def classify_with_phi2(text):
    with open("prompt_training_examples.json") as f:
        examples = json.load(f)

    valid_labels = sorted(set(example["intent"] for example in examples))
    prompt = f"""You are a classification model that assigns multiple relevant INTENTS to a message. Use only these valid labels:\n{', '.join(valid_labels)}
    Message: "{text}"
    Give the INTENT labels only, separated by commas. Do not explain.
    Intents:"""

    
    try:
        response = llm.invoke(prompt)
        return ", ".join([i.strip().upper() for i in response.strip().split(",") if i.strip()])
    except Exception:
        return "UNKNOWN"



def generate_text_response(text, intent, chat_history):
    lang = detect(text)
    conversation = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    if lang == "hi":
        prompt = f"आप SOF मैट्रेस ग्राहक सहायता सहायक हैं। अब तक की बातचीत:\n{conversation}\nग्राहक: {text}\nसहायक:"
    else:
        prompt = f"You are a helpful assistant for SOF mattress customers. Here is the chat so far:\n{conversation}\nCustomer: {text}\nAssistant:"

    try:
        return llm.invoke(prompt).strip()
    except Exception:
        return "Sorry, something went wrong while generating a response."

# === Chatbot logic with memory ===
def chatbot(user_msg, history):
    if history is None:
        history = []

    bert_intent = classify_with_bert(user_msg)
    phi_intent = classify_with_phi2(user_msg)
    reply = generate_text_response(user_msg, bert_intent, history)

    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": reply})

    return history, bert_intent, phi_intent, history

# === Gradio UI ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 SOF Mattress Chatbot
    Welcome to the SOF AI assistant! Ask us anything about your mattress orders, EMI, features, and support.
    """, elem_id="header")

    with gr.Row():
        with gr.Column(scale=2):
            chat_ui = gr.Chatbot(label="🛏️ Chat Window", type="messages")
            user_input = gr.Textbox(label="💬 Your message", placeholder="Type your query here and press Enter")
        with gr.Column(scale=1):
            intent_bert = gr.Textbox(label="🤖 Intent (BERT)", interactive=False)
            intent_phi = gr.Textbox(label="🧠 Intent (LLaMA 3.2)", interactive=False)

    state = gr.State([])
    user_input.submit(fn=chatbot, inputs=[user_input, state], outputs=[chat_ui, intent_bert, intent_phi, state])

    gr.Markdown("""
    ---
    🔒 This bot respects your privacy. Built with 💙 by SOF AI Team.
    """, elem_id="footer")


demo.launch()
