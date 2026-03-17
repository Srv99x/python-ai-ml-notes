# Revision Notes:
# Topic: Practice problems and concept reinforcement
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

n = int(input("Enter a number: "))
for i in range(1, n+1):
    print(" "*(n-i), end="")
    print("*"* (2*i-1), end="")
    print("")

for i in range(1, n + 1):
    print(" " * (i - 1), end="")
    print("*" * (2 * (n - i) + 1))

