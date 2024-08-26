#pragma once

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <utility>
#include <random>
#include <omp.h>
#include <arm_neon.h>

using namespace std;

template <class T>
class Matrix {
    size_t myRows, myCols;
    vector<T> myVector;

public:
    // Constructors
    Matrix() : myRows(0), myCols(0) {}
    Matrix(const size_t rows, const size_t cols)  : myRows(rows), myCols(cols), myVector(rows * cols) {
        // Initialize in Parallel for 0
        #pragma omp parallel for
        for (size_t i = 0; i < rows * cols; ++i) {myVector[i] = T(0.0);}
    }
    Matrix(const size_t rows, const size_t cols, const T val) : myRows(rows), myCols(cols), myVector(rows * cols) {
        // Initialize in Parallel for val
        #pragma omp parallel for
        for (size_t i = 0; i < rows * cols; ++i) {myVector[i] = val;}
    }

    // Copy Constructor
    Matrix(const Matrix& rhs) : myRows(rhs.myRows), myCols(rhs.myCols), myVector(rhs.myRows * rhs.myCols) {
        #pragma omp parallel for
        for (size_t i = 0; i < myRows * myCols; ++i) {myVector[i] = rhs.myVector[i];}
    }

    // COpy Assignment
    Matrix& operator=(const Matrix& rhs) {
        if (this == &rhs) return *this;
        myRows = rhs.myRows;
        myCols = rhs.myCols;
        myVector.resize(myRows * myCols);
        
        #pragma omp parallel for
        for (size_t i = 0; i < myRows * myCols; ++i) {myVector[i] = rhs.myVector[i];}
        return *this;
    }

    // Move Constructor
    Matrix(Matrix&& rhs) noexcept : myRows(rhs.myRows), myCols(rhs.myCols), myVector(std::move(rhs.myVector)) {
        rhs.myRows = 0;
        rhs.myCols = 0;
    }

    // Move Assignment Operator
    Matrix& operator=(Matrix&& rhs) noexcept {
        if (this == &rhs) return *this;
        myRows = rhs.myRows;
        myCols = rhs.myCols;
        myVector = std::move(rhs.myVector);
        rhs.myRows = 0;
        rhs.myCols = 0;
        return *this;
    }

    // Size Accessors
    size_t num_rows() const {return myRows;}
    size_t num_cols() const {return myCols;}

    // Element Accessors
    T* operator[](const size_t row) { return &myVector[row * myCols]; }
    const T* operator[](const size_t row) const { return &myVector[row * myCols]; }
    bool empty() const { return myVector.empty(); }

    // Iterators
    typedef typename vector<T>::iterator iterator;
    typedef typename vector<T>::const_iterator const_iterator;
    iterator begin() { return myVector.begin(); }
    iterator end() { return myVector.end(); }
    const_iterator begin() const { return myVector.begin(); }
    const_iterator end() const { return myVector.end(); }

    // Transpose
    Matrix transpose() const {
        Matrix result(myCols, myRows);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {
                result[j][i] = (*this)[i][j];
            }}
        return result;
    }

    // Dot product
    Matrix dot(const Matrix& rhs) const {
        Matrix result(myRows, rhs.num_cols());
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < rhs.num_cols(); ++j) {
                T sum = 0;
                for (size_t k = 0; k < myCols; ++k) {sum += (*this)[i][k] * rhs[k][j];}
                result[i][j] = sum;
            }
        }
        return result;
    }

    Matrix operator+(T scalar) const {
        Matrix result(myRows, myCols);
        #pragma omp parallel for
        for (size_t i = 0; i < myRows * myCols; ++i) {result.myVector[i] = myVector[i] + scalar;}
        return result;
    }

    Matrix operator-(T scalar) const {
        Matrix result(myRows, myCols);
        #pragma omp parallel for
        for (size_t i = 0; i < myRows * myCols; ++i) {result.myVector[i] = myVector[i] - scalar;}
        return result;
    }

    Matrix operator*(T scalar) const {
        Matrix result(myRows, myCols);
        #pragma omp parallel for
        for (size_t i = 0; i < myRows * myCols; ++i) {result.myVector[i] = myVector[i] * scalar;}
        return result;
    }

    Matrix operator/(T scalar) const {
        Matrix result(myRows, myCols);
        #pragma omp parallel for
        for (size_t i = 0; i < myRows * myCols; ++i) {result.myVector[i] = myVector[i] / scalar;}
        return result;
    }

    // Matrix x Vector multiplication
    vector<T> operator*(const vector<T>& vec) const {
        vector<T> result(myRows, T(0));
        #pragma omp parallel for
        for (size_t i = 0; i < myRows; ++i) {
            T sum = 0;
            for (size_t j = 0; j < myCols; ++j) {sum += (*this)[i][j] * vec[j];}
            result[i] = sum;
        }
        return result;
    }

    // Matrix + Vector addition
    Matrix operator+(const vector<T>& vec) const {
        Matrix result(myRows, myCols);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {
                result[i][j] = (*this)[i][j] + vec[i];
            }}
        return result;
    }

    // Matrix - Vector subtraction
    Matrix operator-(const vector<T>& vec) const {
        Matrix result(myRows, myCols);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {
                result[i][j] = (*this)[i][j] - vec[i];
            }}
        return result;
    }

    // Matrix x Matrix multiplication
    Matrix operator*(const Matrix& rhs) const {
        Matrix result(myRows, rhs.num_cols());
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < rhs.num_cols(); ++j) {
                T sum = 0;
                for (size_t k = 0; k < myCols; ++k) {sum += (*this)[i][k] * rhs[k][j];}
                result[i][j] = sum;
            }
        }
        return result;
    }

    // ElementWise Matrix - Matrix subratction
    Matrix operator-(const Matrix& rhs) const {
        Matrix result(myRows, myCols);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {
                result[i][j] = (*this)[i][j] - rhs[i][j];
            }}
        return result;
    }
    
    // Matrix + Matrix addition
    Matrix operator+(const Matrix& rhs) const {
        Matrix result(myRows, myCols);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {
                result[i][j] = (*this)[i][j] + rhs[i][j];
            }}
        return result;
    }

    // Print Matrix to command line
    void print(ostream& os) const {
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {os << (*this)[i][j] << " ";}
            os << endl;
        }
    }


    // Return Dimensions
    pair<size_t, size_t> getDimensions() const {return make_pair(num_rows(), num_cols());}

    // Element-wise multiplication
    Matrix elementwiseMultiply(const Matrix& rhs) const {
        Matrix result(myRows, myCols);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {
                result[i][j] = (*this)[i][j] * rhs[i][j];
            }}
        return result;
    }


    // Sum along rows (resulting in a vector of column sums)
    vector<T> sumAlongRows() const {
        vector<T> result(myRows, T(0));
        #pragma omp parallel for
        for (size_t i = 0; i < myRows; ++i) {
            T sum = 0;
            for (size_t j = 0; j < myCols; ++j) {sum += (*this)[i][j];}
            result[i] = sum;
        }
        return result;
    }
};

// Scalar Multiplication
template <typename T>
vector<T> operator*(const vector<T>& vec, double scalar) {
    vector<T> result(vec.size());
    #pragma omp parallel for
    for (size_t i = 0; i < vec.size(); ++i) {result[i] = vec[i] * scalar;}
    return result;
}

// Vector Addition
template <typename T>
vector<T> operator+(const vector<T>& vec1, const vector<T>& vec2) {
    vector<T> result(vec1.size());
    #pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); ++i) {result[i] = vec1[i] + vec2[i];}
    return result;
}

// Elementwise Vector Addition
template <typename T>
vector<T> operator+(const vector<T>& vec1, const double& x) {
    vector<T> result(vec1.size());
    #pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); ++i) {result[i] = vec1[i] + x;}
    return result;
}

// ElementWise Vector Multiplication
template <typename T>
vector<T> operator*(const vector<T>& vec1, const vector<T>& vec2) {
    vector<T> result(vec1.size());
    #pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); ++i) {result[i] = vec1[i] * vec2[i];}
    return result;
}

template <typename T>
vector<T> operator-(const vector<T>& vec1, const vector<T>& vec2) {
    vector<T> result(vec1.size());
    #pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); ++i) {result[i] = vec1[i] - vec2[i];}
    return result;
}


vector<double> operator/(const vector<double>& a, const vector<double>& b) {
    vector<double> result(a.size());
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); ++i) {result[i] = a[i] / b[i];}
    return result;
}