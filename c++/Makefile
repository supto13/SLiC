
CXX = g++
# CXXFLAGS = -std=c++11 -Wall -Wextra -pedantic -ffast-math -fno-common
CXXFLAGS = -I/usr/include/eigen3 -std=c++11 -Wall -Wextra -pedantic -ffast-math -fno-common
OBJS = slic_compression.o wavelet.o
TARGET = slic_compression
OBJS1 = slic_decompression.o wavelet.o
TARGET1 = slic_decompression

ifeq ($(OS),Windows_NT)
	# TODO
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		CXXFLAGS += -O3
	endif
	ifeq ($(UNAME_S),Darwin)
		CXXFLAGS += -stdlib=libc++ -Ofast
	endif
endif

# Default rule
all: compile compile1

# Dependency rules
slic_compression.o : config.h knncpp.h wavelet.h vector.h
slic_decompression.o : config.h wavelet.h vector.h
wavelet.o : wavelet.h vector.h

# Pattern rule to create an object file from a cpp file
# $@ expands to the target
# $^ expands to the dependencies
%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

compile: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $^

compile1: $(OBJS1)
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $^

clean:
	rm $(TARGET) $(OBJS) $(TARGET1) $(OBJS1)
