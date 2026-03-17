# Revision Notes:
# Topic: Using super() for parent class method access
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

class Car:
    def __init__(self, type):
        self.type = type

    @staticmethod
    def start():
        print("The car has started...")

    @staticmethod
    def stop():
        print("The car has stopped...")

class ToyotaCar(Car):
    def __init__(self, name, type):
        self.name = name
        super().__init__(type)
        super().start()

car1 = ToyotaCar("Prius", "Ev")
print(car1.type)


