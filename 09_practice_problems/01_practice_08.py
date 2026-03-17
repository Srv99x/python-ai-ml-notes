# Revision Notes:
# Topic: Practice problems and concept reinforcement
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

n = int(input("Enter a number: "))

product = 1
for i in range(1, n+1):
    product = product*i
print(f"The factorial of {n} is: {product}")



