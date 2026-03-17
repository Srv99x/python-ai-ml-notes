# Revision Notes:
# Topic: Abstraction (hiding implementation details behind interfaces/classes)
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

class Car:
    def __init__(self):
        self.acc = False
        self.brk = False
        self.clutch = False

    def start(self):
        self.clutch = True
        self.acc = True
        print("The car has started!")

car1 = Car()
car1.start()   #The output will only give the print statement as the unnecesssary details are hidden     


