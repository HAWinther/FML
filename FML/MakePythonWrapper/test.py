import MyLibrary as mylib
import numpy as np

# Call the first C++ function we defined in Main.cpp
mylib.test(123)

# Example of how to pass more complex data from C++ to Python
# We allocate a struct with a double* and an int and provide 
# in the InterfaceFile.i how to extract this in python
data = mylib.getData(10)
for i in range(data.get_n()):
  print( data.get_x(i) )

# Indices are checked
try:
  data.get_x(data.get_n())
  raise RuntimeError("get_x did not check the index")
except IndexError as e:
  print("IndexError:", e)

# Python owns the data (%newobject getData in InterfaceFile.i) so the memory we
# allocated in C++ is freed (by freeData) when the object is deleted
del data

# Pass numpy arrays to C++ and use them there
x = np.linspace(1,3,3)
y = np.linspace(1,5,5)
mylib.getNumpyArray(x,y)
