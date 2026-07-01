class Student:
    def __init__(self, name):
        self.name = name
        self.__grade = 0
    
    def set_grade(self, grade):
        if 0 <= grade <= 100:
            self.__grade = grade
        else:
            print('Grade must be between 0 and 100')

    def get_grade(self):
        return self.__grade
    
    def get_status(self):
        if self.__grade >= 60:
            return "Passed"
        else:
            return "Failed"
        
student = Student('Jay')
student.set_grade(85)
print(student.get_grade())
print(student.get_status())


#protected properties
class Person:
    def __init__(self, name, salary):
        self.name = name
        self._salary = salary

p1 = Person('Jay', 100000)
print(p1.name)
print(p1._salary)