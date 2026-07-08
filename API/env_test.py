from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("OPENWEATHER_API_KEY")
print(f"Key loaded: {key}")