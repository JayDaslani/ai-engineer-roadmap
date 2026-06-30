class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hello! My name is {self.name}.")
        print(f"I am {self.age} years old")

p1 = Person('Jay', 20)
p2 = Person('Smeet', 17)
p1.greet()
p2.greet()