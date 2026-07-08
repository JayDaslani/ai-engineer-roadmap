import requests

url = 'https://pokeapi.co/api/v2/pokemon/pikachu'

response = requests.get(url)

print(f"Status: {response.status_code}")
data = response.json()
print(f"Name: {data['name']}")
print(f"Height: {data['height']}")
print(f"Weight: {data['weight']}")
print(f"Type: {data['types'][0]['type']['name']}")


print("\nStats:")
for stat in data['stats']:
    print(f"  {stat['stat']['name']}: {stat['base_stat']}")