import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_weather(city):
    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        print(f"City : {city}")
        print(f"Temp : {data['main']['temp']}°C")
        print(f"Humidity : {data['main']['humidity']}%")
        print(f"Weather : {data['weather'][0]['description']}")
        print(f"Wind Speed : {data['wind']['speed']}")

    elif response.status_code == 401:
        print("API key is wrong")

    elif response.status_code == 404:
        print("City is not found!")

get_weather("Pune")
get_weather("Mumbai")
get_weather("Surat")
get_weather("Ahmedabad")
get_weather("delhi")
