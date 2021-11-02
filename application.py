import ktrain
from flask import Flask, request, jsonify
from flask_cors import CORS

category = ['Geopolitical', 'Hate speech', 'Personal attack', 'Political', 'Profanity', 'Religious',
            'Sexual harassment']

model = ktrain.load_predictor('CyberbullyingDetection_kt')

application = Flask(__name__, static_url_path="/static",
                    static_folder='static')
CORS(application)


@application.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    comment = json_['content']

    cat = model.predict(comment)
    probas = model.predict_proba(comment)

    return jsonify({cat: str(round(max(probas) * 100, 2))})


@application.route('/')
def static_file():
    return application.send_static_file('index.html')


if __name__ == '__main__':
    application.run()
