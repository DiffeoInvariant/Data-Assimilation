EIGEN_INCLUDE_DIR=/home/diffeoinvariant/eigen

CXXFLAGS=-std=c++17 -Wall -Werror -O3 -march=native -mtune=native -I$(EIGEN_INCLUDE_DIR) 
CXX=clang++

.PHONY: hw7

hw7: hw7.cpp
	$(CXX) hw7.cpp $(CXXFLAGS) -o hw7
