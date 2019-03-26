INCL = -I ./frugally-deep/include -I ./eigen -I ./FunctionalPlus/include -I ./json/include
FLAGS = -Wall -g 
all: predictor



predictor: predictor.o
	g++ $(FLAGS) -o predictor predictor.o $(INCL)

predictor.o: predictor.cpp
	g++ $(FLAGS) -c predictor.cpp $(INCL)

clean: rm *.o
