# Revision Notes:
# Topic: self keyword and instance context in methods
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

class Student:
    #Class attributes
    name = "Sourav"              
    language = "Js"
    rollNo = 14
    sec = 'A'

    def __init__(self):         #Dunder method, it is automatically called when an object is created
        print("I'm creating an object!")

    def getInfo(self):
        print(f"The name  is {self.name}.\nThe language is {self.language}")

    @staticmethod
    def greet():
        print("Good morning")

info = Student()
info.language = "Py"      #object attribute or instance attribute
# print(f"{info.name}\n{info.rollNo}\n{info.sec}\n{info.language}")    

info.getInfo()    #self parameter
# Student.getInfo(info)     #self parameter
info.greet()     
