#pragma once

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <utility>
#include <random>

using namespace std;

template <class T>
class Matrix {
    size_t myRows, myCols;
    vector<T> myVector;

public:
    // Constructors
    Matrix() : myRows(0), myCols(0) {}
    Matrix(const size_t rows, const size_t cols) : myRows(rows), myCols(cols), myVector(rows * cols) {}
    Matrix(const size_t rows, const size_t cols, const T val) : myRows(rows), myCols(cols), myVector(rows * cols, T(val)) {}

    // Size Accessors
    size_t num_rows() const { return myRows; }
    size_t num_cols() const { return myCols; }

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

    // Member functions for dot product, transpose, scalar operations, and matrix multiplication

    // Dot product
    Matrix dot(const Matrix& rhs) const {
        Matrix result(myRows, rhs.num_cols());
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < rhs.num_cols(); ++j) {
                for (size_t k = 0; k < myCols; ++k) {
                    result[i][j] += (*this)[i][k] * rhs[k][j];
                }
            }
        }
        return result;
    }

    // Transpose
    Matrix transpose() const {
        Matrix result(myCols, myRows);
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {
                result[j][i] = (*this)[i][j];
            }}
        return result;
    }

    // Scalar operations
    Matrix operator+(T scalar) const {
        Matrix result(myRows, myCols);
        transform(myVector.begin(), myVector.end(), result.myVector.begin(), [scalar](T val) { return val + scalar; });
        return result;
    }

    Matrix operator-(T scalar) const {
        Matrix result(myRows, myCols);
        transform(myVector.begin(), myVector.end(), result.myVector.begin(), [scalar](T val) { return val - scalar; });
        return result;
    }

    Matrix operator*(T scalar) const {
        Matrix result(myRows, myCols);
        transform(myVector.begin(), myVector.end(), result.myVector.begin(), [scalar](T val) { return val * scalar; });
        return result;
    }

    Matrix operator/(T scalar) const {
        Matrix result(myRows, myCols);
        transform(myVector.begin(), myVector.end(), result.myVector.begin(), [scalar](T val) { return val / scalar; });
        return result;
    }

    // Matrix x Vector multiplication
    vector<T> operator*(const vector<T>& vec) const {
        if (myCols != vec.size()) {throw invalid_argument("Matrix columns must match vector size for multiplication.");}
        vector<T> result(myRows, T(0));
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {
                result[i] += (*this)[i][j] * vec[j];
            }
        }
        return result;
    }

    // Matrix + Vector addition
    Matrix operator+(const vector<T>& vec) const {
        if (myRows != vec.size()) {throw invalid_argument("Matrix rows must match vector size for addition.");}
        Matrix result(myRows, myCols);
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {
                result[i][j] = (*this)[i][j] + vec[i];
            }
        }
        return result;
    }

    // Matrix - Vector subtraction
    Matrix operator-(const vector<T>& vec) const {
        if (myRows != vec.size()) {throw invalid_argument("Matrix rows must match vector size for addition.");}
        Matrix result(myRows, myCols);
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {
                result[i][j] = (*this)[i][j] - vec[i];
            }
        }
        return result;
    }

    // Matrix x Matrix multiplication
    Matrix operator*(const Matrix& rhs) const {
        if (myCols != rhs.num_rows()) {throw invalid_argument("Matrix A columns must match Matrix B rows for multiplication.");}
        Matrix result(myRows, rhs.num_cols());
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < rhs.num_cols(); ++j) {
                for (size_t k = 0; k < myCols; ++k) {
                    result[i][j] += (*this)[i][k] * rhs[k][j];
                }
            }
        }
        return result;
    }

    // ElementWise Matrix - Matrix subratction
    Matrix operator-(const Matrix& rhs) const {
        if (rhs.num_rows() != myRows || rhs.num_cols() != myCols) {throw invalid_argument("Matrix sizes must match");}
        Matrix result(myRows, rhs.num_cols());
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < rhs.num_cols(); ++j) {
                result[i][j] = (*this)[i][j] - rhs[i][j];
            }
        }
        return result;
    }
    
    // Matrix + Matrix addition
    Matrix operator+(const Matrix& rhs) const {
        if (rhs.num_rows() != myRows || rhs.num_cols() != myCols) {throw invalid_argument("Matrix sizes must match");}
        Matrix result(myRows, rhs.num_cols());
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < rhs.num_cols(); ++j) {
                result[i][j] = (*this)[i][j] + rhs[i][j];
            }
        }
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


    // Sum along columns (resulting in a vector of row sums)
    vector<T> sumAlongColumns() const {
        vector<T> result(myCols, T(0));
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {
                result[j] += (*this)[i][j];
            }
        }
        return result;
    }

    // Element-wise multiplication
    Matrix elementwiseMultiply(const Matrix& rhs) const {
        if (myRows != rhs.num_rows() || myCols != rhs.num_cols()) {
            throw invalid_argument("Matrix dimensions must match for element-wise multiplication.");
        }
        Matrix result(myRows, myCols);
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {
                result[i][j] = (*this)[i][j] * rhs[i][j];
            }
        }
        return result;
    }

    // ElementWise Division of Matrix by Another
    Matrix<double> elementwiseDivide(const Matrix<double>& rhs) const {
        if (myRows != rhs.num_rows() || myCols != rhs.num_cols()) {
            throw invalid_argument("Matrix dimensions must match for element-wise division.");
        }
        Matrix<double> result(myRows, myCols);
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {
                result[i][j] = (*this)[i][j] / rhs[i][j];
            }
        }
        return result;
    }

    // ElementWise Application of fuction to Matrix
    Matrix<double> apply(double (*func)(double)) const {
        Matrix<double> result(myRows, myCols);
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {
                result[i][j] = func((*this)[i][j]);
            }
        }
        return result;
    }


    // Sum along rows (resulting in a vector of column sums)
    vector<T> sumAlongRows() const {
        vector<T> result(myRows, T(0));
        for (size_t i = 0; i < myRows; ++i) {
            for (size_t j = 0; j < myCols; ++j) {
                result[i] += (*this)[i][j];
            }
        }
        return result;
    }
};

// Scalar Multiplication
template <typename T>
vector<T> operator*(const vector<T>& vec, double scalar) {
    vector<T> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {result[i] = vec[i] * scalar;}
    return result;
}

// Vector Addition
template <typename T>
vector<T> operator+(const vector<T>& vec1, const vector<T>& vec2) {
    vector<T> result(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {result[i] = vec1[i] + vec2[i];}
    return result;
}

// Elementwise Vector Addition
template <typename T>
vector<T> operator+(const vector<T>& vec1, const double& x) {
    vector<T> result(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {result[i] = vec1[i] + x;}
    return result;
}

// ElementWise Vector Multiplication
template <typename T>
vector<T> operator*(const vector<T>& vec1, const vector<T>& vec2) {
    vector<T> result(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] * vec2[i];
    }
    return result;
}

template <typename T>
vector<T> operator-(const vector<T>& vec1, const vector<T>& vec2) {
    vector<T> result(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {result[i] = vec1[i] - vec2[i];}
    return result;
}


vector<double> operator/(const vector<double>& a, const vector<double>& b) {
    if (a.size() != b.size()) { throw invalid_argument("Vector sizes must match for element-wise division.");}
    vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {result[i] = a[i] / b[i];}
    return result;
}