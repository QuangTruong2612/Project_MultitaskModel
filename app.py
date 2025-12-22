from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
import os
from Project_MultitaskModel.utils.common import decodeImage
from Project_MultitaskModel.pipeline.prediction import PredictPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.predict_pipeline = PredictPipeline(self.filename)
        

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    # os.system("dvc repro")
    return "Training done successfully!"



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.predict_pipeline.predict()
    response_data = [
            {"class": result[0]['class']}, 
            {"image": result[0]['image']}
        ]
    return jsonify(response_data)
 


if __name__ == "__main__":
    clApp = ClientApp()

    app.run(host='0.0.0.0', port=8080) #for AWS