CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2

# Targets
all: test_conv_accelerator

conv_accelerator.o: conv_accelerator.cpp conv_accelerator.h
	$(CXX) $(CXXFLAGS) -c conv_accelerator.cpp -o conv_accelerator.o

test_conv_accelerator.o: test_conv_accelerator.cpp conv_accelerator.h
	$(CXX) $(CXXFLAGS) -c test_conv_accelerator.cpp -o test_conv_accelerator.o

test_conv_accelerator: conv_accelerator.o test_conv_accelerator.o
	$(CXX) $(CXXFLAGS) conv_accelerator.o test_conv_accelerator.o -o test_conv_accelerator

test: test_conv_accelerator
	./test_conv_accelerator

clean:
	rm -f *.o test_conv_accelerator

.PHONY: all test clean
