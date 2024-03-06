from flask import Flask, request, jsonify, render_template_string
import pickle
import os

# Get the current working directory
current_dir = os.getcwd()

# Load the trained model
with open(os.path.join(current_dir, 'Naive Bayes Classifier', 'spam_model.pkl'), 'rb') as f:
    model = pickle.load(f)

# Load the vectorizer
with open(os.path.join(current_dir, 'Naive Bayes Classifier', 'vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

# Create the Flask app
app = Flask(__name__)

# HTML content
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Spam Detection</title>
</head>
<body>
    <h1>Spam Detection</h1>
    <form action="/predict" method="post">
        <textarea name="text" rows="10" cols="40" placeholder="Enter text here to see if its spam or ham (not spam)"></textarea>
        <br>
        <input type="submit" value="Submit">
    </form>
    <div id="prediction"></div>
</body>
</html>
"""

#  route for the home page
@app.route('/', methods=['GET'])
def home():
    return render_template_string(html_content)

#  route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the request
    text = request.form['text']

    # Vectorize the input text
    X = vectorizer.transform([text]).toarray()

    # Make a prediction
    prediction = model.predict(X)[0]

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
