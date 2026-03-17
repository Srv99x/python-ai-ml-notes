# Revision Notes:
# Topic: Practice problems and concept reinforcement
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

def sum(n):
    if(n<=1):
        return 1
    return sum(n-1)+n

print(sum(4)) 

