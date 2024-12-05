from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer
import pandas as pd
from PIL import Image

import base64
from io import BytesIO

import warnings
warnings.filterwarnings("ignore", message="These pretrained weights were trained with QuickGELU")

app = Flask(__name__)

# Load precomputed embeddings and model setup
df = pd.read_pickle("image_embeddings.pickle")
image_dir = "static/coco_images_resized"
model, _, preprocess = create_model_and_transforms("ViT-B/32", pretrained="openai")
tokenizer = get_tokenizer("ViT-B-32")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.form
    query_type = data.get("query_type")
    embeddings = torch.stack([torch.tensor(embed) for embed in df["embedding"]])

    # Handle Image Upload
    if "image_base64" in data:
        image_base64 = data.get("image_base64")
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = preprocess(image).unsqueeze(0)

    # Handle Text Query
    if query_type == "text":
        text_query = data.get("text")
        query_embedding = F.normalize(model.encode_text(tokenizer([text_query]))).squeeze(0)

    # Handle Image Query
    elif query_type == "image":
        query_embedding = F.normalize(model.encode_image(image)).squeeze(0)

    # Handle Hybrid Query
    elif query_type == "hybrid":
        text_query = data.get("text")
        lam = float(data.get("lam"))
        image_embedding = F.normalize(model.encode_image(image)).squeeze(0)
        text_embedding = F.normalize(model.encode_text(tokenizer([text_query]))).squeeze(0)
        query_embedding = F.normalize(lam * text_embedding + (1 - lam) * image_embedding, dim=0)

    # Compute cosine similarities and get top 5 results
    cosine_similarities = torch.matmul(embeddings, query_embedding)
    top_k_indices = torch.topk(cosine_similarities, k=5).indices.tolist()
    results = [
        {
            "file_name": df.iloc[idx]["file_name"],
            "similarity": cosine_similarities[idx].item(),
            "image_url": f"/{image_dir}/{df.iloc[idx]['file_name']}",
        }
        for idx in top_k_indices
    ]

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
