# Revision Notes:
# Topic: Accessing private-like attributes and name mangling in Python
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

class Person:
    def __hello(self):
        print("Hello buddy!")

    def welcome(self):
        self.__hello()

p1 = Person()
print(p1.welcome())
