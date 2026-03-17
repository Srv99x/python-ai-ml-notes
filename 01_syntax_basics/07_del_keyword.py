# Revision Notes:
# Topic: del keyword: deleting variables, items, and attributes
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

class Student: 
    def __init__(self, name):
        self.name = name

s1 = Student("Sourav")
print(s1.name)
del s1.name  #The del keyword deletes the object properties or obj itself
