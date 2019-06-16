all: predictor test_suite predictorslow

test_suite: profile_predictor_test_suite.o profile_predictor.o
	gcc -o test_suite profile_predictor_test_suite.o profile_predictor.o -lm

predictor: main.o profile_predictor.o
	gcc -o predictor main.o profile_predictor.o -lm

predictorslow: mainslow.o profile_predictorslow.o
	gcc -o predictorslow mainslow.o profile_predictorslow.o -lm

clean:
	rm *.o

profile_predictor_test_suite.o: profile_predictor_test_suite.c
	gcc -c -O3 -Wall profile_predictor_test_suite.c

main.o: main.c
	gcc -c -O3 -Wall main.c

profile_predictor.o: profile_predictor.c profile_predictor.h
	gcc -c -O3 -Wall profile_predictor.c

profile_predictorslow.o: profile_predictor.c profile_predictor.h
	gcc -c -g -Wall -o profile_predictorslow.o profile_predictor.c

mainslow.o: main.c
	gcc -c -g -Wall -o mainslow.o main.c
