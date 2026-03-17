# Revision Notes:
# Topic: Practice problems and concept reinforcement
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

#greatest no between 4 numbers
# a1 = int(input("enter number 1: ",))
# a2 = int(input("enter number 2: ",))
# a3 = int(input("enter number 3: ",))
# a4 = int(input("enter number 4: ",))

# if(a1>a2 and a1>a3 and a1>a4):
#     print("greatest number is a1", a1)
# elif(a2>a1 and a2>a3 and a2>a4):
#     print("greatest number is a2", a2)
# elif(a3>a2 and a3>a1 and a3>a4):
#     print("greatest number is a3", a3)
# elif(a4>a2 and a4>a3 and a4>a1):
#     print("greatest number is a4", a4)

#student pass or fail detector
marks1 = int(input("Enter marks for 1st sub: "))
marks2 = int(input("Enter marks for 2nd sub: "))
marks3 = int(input("Enter marks for 3rd sub: "))

total_percentage = (100*(marks1+marks2+marks3))/300

if(total_percentage>=40 and marks1>=33 and marks2>=33 and marks3>=33):
    print("YOU HAVE PASSED YOUR EXAM")
else:
    print("YOU FAILED BITCH! TRY AGAIN IN NEXT LIFE")
