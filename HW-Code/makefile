CC=clang
CXX=clang++

BOOST_DIR=/usr/local/include/boost/

CXXFLAGS=-std=c++17 -O3 -I$(BOOST_DIR)

.PHONY: hw1

hw1: HW1/hw1.cpp
	@echo "Compiling Homework 1 code. This may take a few seconds."
	@$(CXX) $(CXXFLAGS) HW1/hw1.cpp -o HW1/hw1
	@echo "Running program."
	@./HW1/hw1
	@mv hw1.csv HW1/hw1.csv
	@rm -rf HW1/hw1


