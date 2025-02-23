import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer

# Load trained model and classifier head
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class CustomBERT(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads, num_layers=12):
        super(CustomBERT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads), num_layers
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.projection = nn.Linear(vocab_size, hidden_dim)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.encoder_layers(x)
        x = x.mean(dim=1)
        x = self.output_layer(x)
        x = self.projection(x)
        return x

# Load trained model and classifier
checkpoint = torch.load("sbert_model.pth", map_location=device)
hidden_dim = checkpoint["embedding.weight"].shape[1]
num_heads = max(1, hidden_dim // 64)

bert_model = CustomBERT(30522, hidden_dim, num_heads).to(device)
bert_model.load_state_dict(checkpoint)
bert_model.eval()

classifier_head = nn.Linear(hidden_dim * 3, 3).to(device)
classifier_head.load_state_dict(torch.load("classifier_head.pth", map_location=device))
classifier_head.eval()

# Define label mapping
label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

def predict_relationship(sentence1, sentence2):
    tokens1 = tokenizer(sentence1, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    tokens2 = tokenizer(sentence2, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
    input_ids1 = tokens1["input_ids"].to(device)
    input_ids2 = tokens2["input_ids"].to(device)
    
    with torch.no_grad():
        output1 = bert_model(input_ids1)
        output2 = bert_model(input_ids2)
        combined = torch.cat([output1, output2, torch.abs(output1 - output2)], dim=1)
        outputs = classifier_head(combined)
        prediction = torch.argmax(outputs, dim=1).item()
    
    return label_map[prediction]

# Streamlit UI
st.set_page_config(page_title="NLI Predictor", page_icon="🔍", layout="centered")
st.title("🔍 Natural Language Inference (NLI) Predictor")
st.markdown("Enter two sentences and find out their relationship (Entailment, Contradiction, or Neutral)")

# Input fields
sentence1 = st.text_area("Enter First Sentence", "The sky is blue.")
sentence2 = st.text_area("Enter Second Sentence", "The ocean is blue.")

# Predict button
if st.button("🔮 Predict Relationship"):
    with st.spinner("Analyzing... 🚀"):
        prediction = predict_relationship(sentence1, sentence2)
    st.success(f"**Prediction: {prediction}**")
    
# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & PyTorch. Created by Ponkrit Kaewsawee ST124960")
