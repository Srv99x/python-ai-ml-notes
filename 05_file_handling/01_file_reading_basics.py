# Revision Notes:
# Topic: File handling basics: open, read, and close
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

f = open("text.txt")
data = f.read()
print(data)
f.close()
