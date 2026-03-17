# Revision Notes:
# Topic: Practice problems and concept reinforcement
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.



def greatest(a,b,c):
    if(a>b and a>c):
        print(f"{a} is greatest!")
    elif(b>a and b>c):
        print(f"{b} is greatest!")
    elif(c>b and c>a):
        print(f"{c} is greatest!")

a = int(input("Enter the 1st no: "))
b = int(input("Enter the 2nd no: "))
c = int(input("Enter the 3rd no: "))

greatest(a,b,c)
    
