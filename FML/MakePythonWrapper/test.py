import MyLibrary as mylib
import numpy as np

# Call the first C++ function we defined in Main.cpp
mylib.test(123)

# Example of how to pass more complex data from C++ to Python
# We allocate a struct with a double* and an int and provide 
# in the InterfaceFile.i how to extract this in python
data = mylib.getData(10)
for i in range(10):
  print( data.get_x(i) )

# Free up memory we allocated in C++ (if you feel like it)
mylib.freeData(data)

# Pass numpy arrays to C++ and use them there
x = np.linspace(1,3,3)
y = np.linspace(1,5,5)
mylib.getNumpyArray(x,y)

