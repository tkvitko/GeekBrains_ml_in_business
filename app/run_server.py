import dill
import flask
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
    # Дефолтный ответ API на случай внутренних проблем
    data = {"success": False}

    if flask.request.method == "POST":
        request_json = flask.request.get_json()

        # Список параметров, которые готовы получить
        params_list = {
            "Location": None,
            "MinTemp": None,
            "MaxTemp": None,
            "Rainfall": None,
            "Evaporation": None,
            "Sunshine": None,
            "WindGustDir": None,
            "WindGustSpeed": None,
            "WindDir9am": None,
            "WindDir3pm": None,
            "WindSpeed9am": None,
            "WindSpeed3pm": None,
            "Humidity9am": None,
            "Humidity3pm": None,
            "Pressure9am": None,
            "Pressure3pm": None,
            "Cloud9am": None,
            "Cloud3pm": None,
            "Temp9am": None,
            "Temp3pm": None,
            "RainToday": None
        }

        # Наполнение списка полученными параметрами для скармливания модели
        for param in params_list.keys():
            if request_json[param]:
                params_list[param] = [request_json[param]]

        try:
            # Пробуем предсказать и вернуть ответ
            predictions = model.predict_proba(pd.DataFrame(params_list))
            data["success"] = True
            data["predictions"] = predictions[:, 1][0]

        except Exception as e:
            # Если что-то пошло не так, выведем в консоль исключение
            print(e)

    return flask.jsonify(data)


if __name__ == "__main__":
    # modelpath = "./models/logreg_pipeline.dill"
    modelpath = "./models/weather_catboost.dill"
    load_model(modelpath)

    app.run()

    # Тестирование без API:
    # test_params = {
    #     "Location": ["Albury"],
    #     "MinTemp": [13.4],
    #     "MaxTemp": [22.9],
    #     "Rainfall": [0.6],
    #     "Evaporation": [1.0],
    #     "Sunshine": [1.0],
    #     "WindGustDir": ["W"],
    #     "WindGustSpeed": [44.0],
    #     "WindDir9am": ["W"],
    #     "WindDir3pm": ["WNW"],
    #     "WindSpeed9am": [20.0],
    #     "WindSpeed3pm": [24.0],
    #     "Humidity9am": [71.0],
    #     "Humidity3pm": [22.0],
    #     "Pressure9am": [1007.7],
    #     "Pressure3pm": [1007.1],
    #     "Cloud9am": [8.0],
    #     "Cloud3pm": [7.0],
    #     "Temp9am": [16.9],
    #     "Temp3pm": [21.8],
    #     "RainToday": ["Yes"]
    # }
    # predictions = model.predict_proba(pd.DataFrame(test_params))
    # print(predictions[:, 1][0])
