# Revision Notes:
# Topic: Tuples: immutability, packing, and unpacking
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

#tuples are immutable same as strs

tup = (2,4,5,8,6,9,4)
print(type(tup))
print(tup[0])
print(tup[4])

tup2 = (1,)   #if we miss the "," py will take it as a int
print(tup2)
print(type(tup2)) 
print(tup[1:3])

print(tup.index(4))  #returns the index
print(tup.count(4))    #counts the no of elements

