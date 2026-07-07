import requests

url = "https://jsonplaceholder.typicode.com/posts"

response = requests.get(url)

print(f"Status: {response.status_code}")

data = response.json()

titles = []

for i in range(5):
    titles.append(data[i]['title'])

print(titles)
