import json
import os

class Student:

    def __init__(self, name, roll_number):
        self.name = name
        self.roll_number = roll_number
        self.marks = []

    def add_marks(self, subject, marks):
        self.marks.append({
            'subject' : subject,
            'marks' : marks
        })

        print(f"{subject} mark added for {self.name}")

    def calculate_average(self):
        if len(self.marks) == 0:
            return 0
        
        total = 0

        for mark in self.marks:
            total += mark['marks']
        average = total / len(self.marks)

        return round(average,2)
    
    def display_info(self):
        print('----Student_Info----')
        print(f"Name : {self.name}")
        print(f"Roll_number : {self.roll_number}")
        print("Marks")
        for mark in self.marks:
            print(f" {mark['subject']} : {mark['marks']}")

        print(f"Average : {self.calculate_average()}")
        print(f"--------------------")

    def to_dict(self):
        return {
            "name" : self.name,
            "roll_number" : self.roll_number,
            "marks" : self.marks,
            "average" : self.calculate_average()
        }
    

class StudentManager:

    def __init__(self):
        self.students = []
        self.file_path = 'data/student.json'
        self.load_from_file()

    def add_student(self, name, roll_number):

        for student in self.students:
            if student.roll_number == roll_number:
                print(f"Roll number : {roll_number} already exists!")
                return
            
        new_student = Student(name, roll_number)
        self.students.append(new_student)
        print(f"Student {name} added successfully!")

        self.save_to_file()

    def find_student(self, roll_number):
        for student in self.students:
            if student.roll_number == roll_number:
                return student
        return None
    
    def add_marks(self, roll_number, subject, marks):
        student = self.find_student(roll_number)

        if student is None:
            print(f"Student with roll {roll_number} not found")
            return 
        student.add_marks(subject, marks)
        self.save_to_file()

    def display_all_student(self):
        if len(self.students) == 0:
            print(f"No student found")
            return
        
        print("All student")
        for student in self.students:
            student.display_info()

    def display_student(self, roll_number):
        student = self.find_student(roll_number)
        
        if student is None:
            print(f"Student is not found")
            return
        
        student.display_info()

    def save_to_file(self):

        os.makedirs("data", exist_ok=True)

        data = []
        for student in self.students:
            data.append(student.to_dict())

        with open(self.file_path, "w") as file:
            json.dump(data, file, indent=4)

        print("Data saved succesfully")

    def load_from_file(self):
        # Rule 1: Agar file exist hi nahi karti, toh return kar jao
        if not os.path.exists(self.file_path):
            return
        
        # Rule 2: Agar file exist karti hai par bilkul khali (0 bytes) hai, toh bypass karo
        if os.path.getsize(self.file_path) == 0:
            print("File is empty, starting with a clean slate.")
            return

        try:
            with open(self.file_path, "r") as file:
                data = json.load(file)

            for student_data in data:
                student = Student(
                    student_data['name'],
                    student_data['roll_number']
                )
                student.marks = student_data['marks']
                self.students.append(student)

            print(f"Loaded {len(self.students)} students from file")
            
        except json.JSONDecodeError:
            
            print("[Warning] student.json contains corrupt data. Initializing empty list.")
            self.students = []

def main():
    print("Student Managment System")

    manager = StudentManager()

    while True:
        print("--Menu--")
        print("1. Add Student")
        print("2. Add Marks")
        print("3. View Student")
        print("4. View All Student")
        print("5. Exist")

        choice = input("Enter choice (1-5) : ")

        if choice == '1':
            name = input("Enter Student Name: ")
            roll = input("Enter Roll Number: ")
            manager.add_student(name, roll)

        elif choice == '2':
            roll = input("Enter roll number: ")
            subject = input("Enter subject: ")
            marks = float(input("Enter marks: "))
            manager.add_marks(roll, subject, marks)

        elif choice == '3':
            roll = input('Enter roll number : ')
            manager.display_student(roll)

        elif choice == '4':
            manager.display_all_student()

        elif choice == '5':
            print('Goodbye!')
            break
        
        else:
            print("Invalid choice! Try again.")

if __name__ == "__main__":
    main()



