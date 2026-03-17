# Revision Notes:
# Topic: Functions: parameters, return values, scope
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

#functions are used for code reusability
# def greet():                           #func definition
#     print("hello world")

# def avg():                             #func definition
#     a = int(input("enter no 1: "))
#     b = int(input("enter no 2: "))
#     c = int(input("enter no 3: "))

#     average = (a+b+c)/3
#     print(average)

# greet()                           #function call                       
# avg()                             #function call

#parameterised functions
def greet(name, ending="Bitch"):
    print("Good morning,"+" "+name)
    print(ending)
    return "Thank you!"

a = greet("Dibakar", "Neiga")
b = greet("Sourav")
print(a)
