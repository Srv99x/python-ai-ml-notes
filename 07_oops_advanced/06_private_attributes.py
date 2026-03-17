# Revision Notes:
# Topic: Private/protected naming conventions in classes
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

class Bank:
    def __init__(self, name, balance, accNo, accPass):
        #Accessible - public info
        self.name = name
        self.balance = balance
        #Not-accessible - private info 
        self.__accNo = accNo
        self.__accPass = accPass

A1 = Bank("srv", 70000, 878731, 55556)
print(A1.name, A1.balance)

        
