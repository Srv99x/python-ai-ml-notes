# Revision Notes:
# Topic: Properties with @property getter/setter patterns
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

class Student:
    def __init__(self, phy, chem, math):
        self.phy = phy
        self.chem = chem 
        self.math = math

    @property
    def calcPercentage(self):
        return str((self.phy + self.chem + self.math)/3) + "%"
    
stu1 = Student(98,89,95)
print(stu1.calcPercentage)
stu1.phy = 70
print(stu1.calcPercentage)

