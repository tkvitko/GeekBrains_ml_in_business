import flask
import dill
import pandas as pd


dill._dill._reverse_typemap['ClassType'] = type


app = flask.Flask(__name__)
model = None


def load_model(model_path):
    # load the pre-trained model
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)


@app.route("/", methods=["GET"])
def general():
    return "Welcome to fraudelent prediction process"


@app.route("/predict", methods=["POST"])
def predict():

    data = {"success": False}

    if flask.request.method == "POST":
        description, company_profile, benefits = "", "", "qq"
        request_json = flask.request.get_json()
        if request_json["description"]:
            description = request_json['description']
        if request_json["company_profile"]:
            company_profile = request_json['company_profile']
        if request_json["benefits"]:
            benefits = request_json['benefits']

        try:
            predictions = model.predict_proba(pd.DataFrame({"description": [description],
                                                      "company_profile": [company_profile],
                                                      "benefits": [benefits]}))
            data["success"] = True
            data["predictions"] = predictions[:, 1][0]
            data["description"] = description

        except Exception as e:
            print(e)

    return flask.jsonify(data)


if __name__ == "__main__":

    modelpath = "./models/logreg_pipeline.dill"
    load_model(modelpath)

    app.run()
