# Revision Notes:
# Topic: Dictionary creation, access, update, and iteration
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

#dictionaries are used to store data values in key:value pairs, they are unordered, mutable and dont allow duplicate keys 
info = {
    # "key" : "value"
    "name" : "sourav",
    "age" : 19,
    "marks" : [90, 99, 98],
    "is_adult" : True,
    "subjects" : ["py", "Cpp", "html", "js" ],
    "topics" : ("OOPs", "loopps", "conditions"),
    "CGPA" : 9.8
}

# print(info["name"])
# print(info["subjects"])

info["name"] = "srv"  #overwrite
info["surname"] = "chakraborty"

# print(info)
 
null_dict = {}
null_dict["name"] = "sourav"
# print(null_dict)

#nested dictionary
student = {
    "name" : "sourav",
    "subMarks" : {
        "maths" : 96,
        "chem" : 99,
        "phy" : 98
    }
}

# print(student)
# print(student["subMarks"]["chem"])

print(list(student.keys()))
print(len(student))
print(list(student.values()))
