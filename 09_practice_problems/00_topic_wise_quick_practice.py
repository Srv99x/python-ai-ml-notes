# Revision Notes:
# Topic: Mixed quick practice from syntax to OOP
# - Try each task first by yourself, then compare with sample approach.
# - Uncomment one block at a time and run.

# Task 1: Syntax + Input
# Take two numbers and print sum, difference, product.
# a = int(input("Enter first number: "))
# b = int(input("Enter second number: "))
# print("sum:", a + b)
# print("difference:", a - b)
# print("product:", a * b)

# Task 2: Conditionals
# Check whether number is positive, negative, or zero.
# n = int(input("Enter a number: "))
# if n > 0:
#     print("Positive")
# elif n < 0:
#     print("Negative")
# else:
#     print("Zero")

# Task 3: Loops + List
# Print squares of numbers from 1 to 5.
# nums = [1, 2, 3, 4, 5]
# for x in nums:
#     print(x, "->", x * x)

# Task 4: Functions
# Create a function that returns factorial.
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print("factorial(5) =", factorial(5))

# Task 5: OOP Basics
# Create a Student class and print details.
class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def show(self):
        print(f"Student: {self.name}, Marks: {self.marks}")

s1 = Student("Sourav", 92)
s1.show()

# Task 6: Inheritance
class Animal:
    def speak(self):
        print("Animal sound")

class Dog(Animal):
    def speak(self):
        print("Bark")

d = Dog()
d.speak()
