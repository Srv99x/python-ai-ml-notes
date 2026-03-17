# Revision Notes:
# Topic: OOP fundamentals: classes, objects, methods
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

class Student:
    #Class attributes
    name = "Sourav"              
    dept = "BTech CSE"
    rollNo = 14
    sec = 'A'

info = Student()
info.language = "Py"      #object attribute or instance attribute
print(f"{info.name}\n{info.dept}\n{info.rollNo}\n{info.sec}\n{info.language}")


