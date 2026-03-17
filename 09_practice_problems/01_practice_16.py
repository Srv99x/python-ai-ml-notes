# Revision Notes:
# Topic: Practice problems and concept reinforcement
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def get_sum(self):
        sum = 0
        for val in self.marks:
            sum += val
        print(F"Hello {self.name} your average marks is {sum/3}")

s1 = Student("Sourav", [78, 89, 97])
s1.get_sum()

