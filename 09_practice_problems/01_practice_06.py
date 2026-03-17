# Revision Notes:
# Topic: Practice problems and concept reinforcement
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

n = int(input("Enter a number:"))

for i in range(2, n):
    if(n%i==0):
        print("Number is not prime")
        break
else:
    print("Number is prime")

