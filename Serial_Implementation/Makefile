# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -std=c++20 -O3

# Target executable
TARGET = test_main

# Source files
SRCS = test_main.cc

# Header files
HDRS = ActivationFuncs.h Initialization.h Matrix.h Network.h

# Object files
OBJS = $(SRCS:.cc=.o)

# Build the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Compile .cc files to .o files
%.o: %.cc $(HDRS)
	$(CXX) $(CXXFLAGS) -c $<

# Clean up the build
clean:
	rm -f $(TARGET) $(OBJS)