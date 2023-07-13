from flask import Flask, request, jsonify

from Three import is_toxic

app = Flask(__name__)


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    sentence = data['sentence']

    # Here, use your existing code to analyze the sentence.
    # Let's assume you have a function called is_toxic that takes a sentence and returns a boolean.
    result = is_toxic(sentence)

    return jsonify({
        'sentence': sentence,
        'is_toxic': result
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True)
