# Revision Notes:
# Topic: Constructors and object initialization with __init__
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

class Employee:
    #Default constructor
    def __init_subclass__(self):
        pass

    #parameterized constructor
    def __init__(self, name, salary, language):         #Dunder method, it is automatically called when an object is created
        self.name =name  #The self parameter is a reference to the current instance of the class, and is used to access variables thst belongs to the class
        self.salary = salary
        self.language = language
        print("I'm creating an object")

    def welcome(self):         #Methods
        print("Hello Employee!", self.name)

    def get_salary(self):
        return self.salary

e1 = Employee("Harry", 130000, "Java")
# Harry.name = "Harry"
print(f"Name: {e1.name}\nLanguage: {e1.language}\nSalary: {e1.salary}")
e1.welcome()
print(f"The salary is: {e1.get_salary()}")

e2 = Employee("Rohan", 135000, "C++")
print(f"Name: {e2.name}\nLanguage: {e2.language}\nSalary {e2.salary}")
e2.welcome()
print(f"The salary is: {e2.get_salary()}")
