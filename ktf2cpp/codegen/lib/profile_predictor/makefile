all: predictor

predictor: main.o profile_predictor.o
	gcc -o predictor main.o profile_predictor.o -lm

main.o: main.c
	gcc -c -g -Wall main.c

profile_predictor.o: profile_predictor.c
	gcc -c -g -Wall profile_predictor.c
