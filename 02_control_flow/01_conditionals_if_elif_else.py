# Revision Notes:
# Topic: Conditional statements: if, elif, else
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

#if-elif-else syntax 
# if(condition):
#     statement1
# elif(condtion):
#     statement2
# else:
#     statementN

print("------Voting calculator------")
age = int(input("enter your age:"))

#if-elif-else ladder
if(age>=18):
    print("you're eligible to vote")
elif(age<0):
    print("Pagla gye ho janab!")
elif(age==0):
    print("You have entered 0")
else:
    print("You aren't eligible")


print("End of program")
