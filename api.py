import pickle

import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

category = ['Body shaming', 'Geopolitical', 'Hate speech', 'Political', 'Profanity', 'Religious', 'Sexual harassment',
            'Troll']

model = load_model(
    "C:\\Users\\ahnaf\\PycharmProjects\\Test-1\\CyberbullyingDetection.h5")
# model.summary()

with open('C:\\Users\\ahnaf\\PycharmProjects\\Test-1\\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    comm = [json_['content']]

    sentence = comm
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=1005, padding='post', truncating='post')
    prediction_main = model.predict(padded)

    i = 0
    for k in range(len(prediction_main[0])):
        print(category[i] + ' = %f %%' % (prediction_main[0, k] * 100))
        i = i + 1

    result = (np.where(prediction_main == max(prediction_main[0])))

    return jsonify({str(category[(result[1][0])]): str(round(max(prediction_main[0]) * 100, 2))})


if __name__ == '__main__':
    app.run(debug=True)
