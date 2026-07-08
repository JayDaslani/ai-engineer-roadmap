import requests

from requests.exceptions import (
    ConnectionError,
    Timeout,
    RequestException
)

def check_status(url):
    try:
        response = requests.get(url)

        if response.status_code == 200:
          print("Success")
          return response.json()
    
        elif response.status_code == 404:
            print("Not Found")
            return None
    
        elif response.status_code == 401:
            print("API key is wrong")
            return None
    
        elif response.status_code == 500:
            print("Server Error!")
            return None
    
        else:
            print(f"Unkown : {response.status_code}")
            return None
    
    except ConnectionError:
        print("Internet not available")
        return None
    
    except Timeout:
        print("Server is slow - timeout")
        return None
    
    except RequestException as e:
        print(f"Error Happend : {e}")
        return None
    
    
print("Right URL")
data = check_status("https://jsonplaceholder.typicode.com/posts/1")

if data:
    print(f"Title: {data['title']}")


print("Wrong ID")
check_status(
    "https://jsonplaceholder.typicode.com/posts/999"
)

print("Wrong Domain")
check_status(
    "https://ye-exist-nahi-karta.com/api"
)

print("Timeout Test")
check_status(
    "https://httpbin.org/delay/10"
)


