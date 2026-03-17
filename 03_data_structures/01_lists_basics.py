# Revision Notes:
# Topic: List operations, methods, slicing, and traversal
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

#List=used to store multiple types of data in a single variable (lists can be changed whereas strs can't be changed)
# marks = [39, 48.0, 34.9, 56, 89.99]
# print(marks)
# print(type(marks))
# print(marks[0], marks[1])
# print(len(marks))
# print(marks[0:5])
# print(marks[-5:-2])

# identity = ["sourav", 18]
# print(identity)

list = [1,9,3,4,5,6,8,7]
# list.append(10) #adds one more 
# list.sort()   #arranges in ascending order
# print(list)
list2 = [1,9,3,4,5,6,8,2]
# list2.sort(reverse=True) #arranges in descending order
# print(list2)

# list.reverse()
# list2.reverse()

# print(list)
# print(list2)

# list.insert(1,2)  #inserts element at index (idx, ele)
# print(list)

list3 = [1,3,5,7]
list3.remove(3)
list3.pop(2)
print(list3)
print(type(list))
