# Revision Notes:
# Topic: Recursion: base case and recursive case design
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

def factorial(n):
    if(n==0 or n==1):
        return 1
    return n*factorial(n-1)

n = int(input("Enter a number to find it's factorial:"))
print(f"The factorial is: {factorial(n)}")
