import requests
import os
import json
from dotenv import load_dotenv
from requests.exceptions import ConnectionError, Timeout

load_dotenv()

def get_weather(city):
    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q" : city,
        "appid" : api_key,
        "units" : "metric"
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()

            weather_data = {
                "city": city,
                "temperature": data['main']['temp'],
                "humidity": data['main']['humidity'],
                "description": data['weather'][0]['description'],
                "wind_speed": data['wind']['speed']
            }
            return weather_data
        
        elif response.status_code == 404:
            print(f'City {city} is not found')
            return None
        
        elif response.status_code == 401:
            print("API key is wrong")

    except Exception as e:
        print(f"Error : {e}")
        return None
    

def save_to_json(weather_data):
    file_path = "API/weather_history.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            history = json.load(f)
    else:
        history = []


    history.append(weather_data)

    with open(file_path, "w") as f:
        json.dump(history, f, indent=4)

    print("Data saved!")

def show_history():
    file_path = "API/weather_history.json"

    if not os.path.exists(file_path):
        print("History is not available")
        return 

    with open(file_path, "r") as f:
        history = json.load(f)

    print(f"Weather History ({len(history)} searches)")
    for item in history:
        print(f"City : {item['city']}")
        print(f"Temp : {item['temperature']}°C")
        print(f"Humidity : {item['humidity']}%")
        print(f"Weather: {item['description']}")
        
def display_weather(data):
    print(f"\n{'='*35}")
    print(f"  {data['city'].upper()} WEATHER")
    print(f"{'='*35}")
    print(f"Temperature : {data['temperature']}°C")
    print(f"Humidity    : {data['humidity']}%")
    print(f"Weather     : {data['description'].title()}")
    print(f"Wind Speed  : {data['wind_speed']} m/s")
    print(f"{'='*35}")

def main():
    print("=== Weather APP ===")

    while True:
        print("1. Check the weather for the city")
        print("2. Chek History")
        print("3. Exit")

        choice = input("Choice (1-3) : ")

        if choice == "1":
            city = input("City name : ")
            data = get_weather(city)

            if data:
                display_weather(data)
                save_to_json(data)

        elif choice == "2":
            show_history()

        elif choice == "3":
            print("Bye")
            break

        else:
            print("Wrong choice!")

if __name__ == "__main__":
    main()
        
