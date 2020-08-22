#include <iostream>

struct Data {
    int n;
    double * x;
};

void test(int n) { std::cout << "Hello " << n << "\n"; }

// Allocate some data and return it to python
Data * getData(int n) {
    Data * a = new Data;
    a->n = n;
    a->x = new double[n];
    for (int i = 0; i < n; i++)
        a->x[i] = i;
    return a;
}

// Free up the memory we have allocated
void freeData(Data * f) {
    std::cout << "Freeing up memory\n";
    if (f->n > 0)
        delete[] f->x;
    delete f;
}

// How to pass (1D) numpy arrays to C++ from python
void getNumpyArray(int nx, double * x, int ny, double * y) {
    for (int i = 0; i < nx; i++)
        std::cout << "x[" << i << "] = " << x[i] << "\n";
    for (int i = 0; i < ny; i++)
        std::cout << "y[" << i << "] = " << y[i] << "\n";
}
