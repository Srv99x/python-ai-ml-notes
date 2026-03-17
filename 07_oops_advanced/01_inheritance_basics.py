# Revision Notes:
# Topic: Single inheritance and method reuse
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

class Car:
    @staticmethod
    def start():
        print("The car has started...")

    @staticmethod
    def stop():
        print("The car has stopped...")

class ToyotaCar(Car):
    def __init__(self, brand):
        self.brand = brand

class Fortuner(ToyotaCar):
    def __init__(self, type):
        self.type = type

car1 = Fortuner("Diesel")
car1.start()
print("Type", car1.type)
# print(car1.brand)
car1.stop()

        
        
