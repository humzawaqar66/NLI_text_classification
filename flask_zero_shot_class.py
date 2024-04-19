from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# this is a flask app
model = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
bart_pipeline = pipeline("zero-shot-classification", model=model)

@app.route('/', methods=['GET','POST'])
def home():
    # Get input data from the request
    data = request.json

    # Extract the sequence from the request data
    input_query = data.get("res")

    # Classify the sequence using the BART model
    result = bart_pipeline(input_query, candidate_labels=["sales", "not sales"])

    # Find the label with the highest score
    max_score_index = result["scores"].index(max(result["scores"]))
    max_score_label = result["labels"][max_score_index]

    # Return the label with the highest score
    response_data = {"label": max_score_label}

    # Return the response in JSON format
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
