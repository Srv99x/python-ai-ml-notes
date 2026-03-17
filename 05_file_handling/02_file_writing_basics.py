# Revision Notes:
# Topic: Writing and appending data to files safely
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

str = "Sourav is a good boy!"
f = open("myfile.txt", "w")
f.write(str)
f.close()
