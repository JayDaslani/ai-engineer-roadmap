class Person:
    def __init__(self, fname, lname):
        self.fname = fname
        self.lname = lname

    def printname(self):
        print(self.fname, self.lname)

class Student(Person):
    def __init__(self, fname, lname, year):
        super().__init__(fname, lname)
        self.graduationyear = 2019

    def welcome(self):
        print(f"Welcome {self.fname} {self.lname} to the class of {self.graduationyear}")



p1 = Student('Jay', 'Dasalani', 2019)
p1.welcome()