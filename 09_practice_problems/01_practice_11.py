# Revision Notes:
# Topic: Practice problems and concept reinforcement
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

def f_to_c(f):
    return 5*(f-32)/9

f = int(input("Enter temparature in F: "))
c = f_to_c(f)
print(f"{round(c,2)} degree celsius")
