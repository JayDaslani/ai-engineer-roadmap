import requests

url = "https://jsonplaceholder.typicode.com/posts"

new_post = {
    "titel": "My first post",
    "body": "I understand post",
    "userId": 1
}

response = requests.post(url, json=new_post)

print(f"Status : {response.status_code}")
print(response.json())