# Revision Notes:
# Topic: Class methods and @classmethod usage
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

class Person:
    name = "Anonymous"

    # def changeName(self, name):
        # Person.name = name
        # self.__class__.name = "RAhul"
    
    @classmethod
    def changeName(cls, name):
        cls.name = name
        
P1 = Person()
P1.changeName("RAhul k")
print(P1.name)
print(Person.name)
