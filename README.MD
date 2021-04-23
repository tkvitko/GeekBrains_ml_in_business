**О проекте**

Тренировочный датасет: https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
Он содержит результаты 10 лет ежедневных наблюдений за погодой из разных мест Австралии.
Передав значения параметров погоды сегодня, можно узнать вероятность того, что дождь будет завтра.

**Запуск сервера**

python3 app/run_server.py

**Пример запроса**

POST http://127.0.0.1:5000/predict
{
    "Location": "Albury",
    "MinTemp": 13.4,
    "MaxTemp": 22.9,
    "Rainfall": 0.6,
    "Evaporation": 1.0,
    "Sunshine": 1.0,
    "WindGustDir": "W",
    "WindGustSpeed": 44.0,
    "WindDir9am": "W",
    "WindDir3pm": "WNW",
    "WindSpeed9am": 20.0,
    "WindSpeed3pm": 24.0,
    "Humidity9am": 71.0,
    "Humidity3pm": 22.0,
    "Pressure9am": 1007.7,
    "Pressure3pm": 1007.1,
    "Cloud9am": 8.0,
    "Cloud3pm": 7.0,
    "Temp9am": 16.9,
    "Temp3pm": 21.8,
    "RainToday": "No"
}