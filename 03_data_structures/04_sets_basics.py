# Revision Notes:
# Topic: Set basics: uniqueness, add/remove, membership
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

# s = {}  #it is an empty dictionary
# s1 = {1,2,6,7,88,90, "Sourav", 34.66, 'A'}  #it is a set 
# e = set()  #this ia an empty set

# # print(type(s), type(s1), type(e))

# print(s1, type(s1))
# s1.add(100)
# print(s1)

#sets are unordered unindexed immutable and cannot have duplicate values  
s = set()
s.add(5)
s.add(1)
s.add(10)
s.add(3)

print(s)

s2 = {10, 3, 50, 1, 7,2, 99, 2}
print(s2)

s2.remove(2)
print(s2)


# A set in Python:
# does NOT store elements in insertion order
# stores elements using hashing
# prints elements in an order that depends on:
    # hash values
    # Python version
    # memory layout
    # internal optimizations

#the order of a set can be different for each versions of py


