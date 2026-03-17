# Revision Notes:
# Topic: Practice problems and concept reinforcement
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

f = open("poem.txt")
content = f.read()
if("Twinkle" in content):
    print("The word twinkle is present in the content!")
else:
    print("The word twinkle is not present in the content!")

f.close()

