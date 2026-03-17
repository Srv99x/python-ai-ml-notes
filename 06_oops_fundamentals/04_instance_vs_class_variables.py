# Revision Notes:
# Topic: Instance variables vs class variables
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

#Object/instance attribute has more presidence than class attribute
class Employee:
    language = "Python"
    name = "sourav"

info = Employee()
info.language = "JavaScript"
print(info.language, info.name)
