
#include "SelfRLCMA.h"

#include "Hex.hpp"

//benchmark functions
#include <shark/ObjectiveFunctions/Benchmarks/Sphere.h>
#include <vector>

#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConvolutionalModel.h>//single convolutional layer
#include <shark/Models/ConcatenatedModel.h>//for stacking layers, provides operator>>
/*
void bluewin(Hex::Game game) {
    game.printhex();
    game.take_turn("A1");
    game.take_turn("A11");
    game.printhex();
    game.take_turn("E5");
    game.take_turn("K1");
    game.printhex();
    game.take_turn("B1");
    game.take_turn("B11");
    game.printhex();
    game.take_turn("C1");
    game.take_turn("C11");
    game.printhex();
    game.take_turn("D1");
    game.take_turn("D11");
    game.printhex();
    game.take_turn("E1");
    game.take_turn("E11");
    game.printhex();
    game.take_turn("F1");
    game.take_turn("F11");
    game.printhex();
    game.take_turn("G1");
    game.take_turn("G11");
    game.printhex();
    game.take_turn("H1");
    game.take_turn("H11");
    game.printhex();
    game.take_turn("I1");
    game.take_turn("I11");
    game.printhex();
    game.take_turn("J1");
    game.take_turn("J11");
    game.printhex();
}*/
#include <stdlib.h>
#include <random>
#include <unistd.h>


void random_strategy (Hex::Game game) {

    std::random_device rd;
    std::mt19937 rng_engine(rd());
    std::uniform_int_distribution<int> dist(1,game.BOARD_SIZE);

    bool won;
    bool valid_move = true;
    std::string random_tile = "";

    char letters[game.BOARD_SIZE];

    for (char i = 65; i < 65+game.BOARD_SIZE; i++) {
        letters[i-65] = i;
    }

    int turn_count = 0;
    while (1) {
        do {
            if (!valid_move) {
                std::cout << "Invalid move!" << std::endl;
            }
            auto randint1 = dist(rng_engine);
            auto randint2 = dist(rng_engine);
            random_tile = letters[randint1-1] + std::to_string(randint2);
            //std::cout << randint1 << ", " << randint2 << std::endl;
            //std::cout << random_tile << std::endl;
        }
        while (!(valid_move = game.take_turn(random_tile, &won)));
        game.printhex();
        //usleep(250000);
        //game.print_segments();
        if (won) {
            std::cout << game.CurrentPlayer() << " LOST" << std::endl;
            break;
        }
        turn_count++;
        if (turn_count >= game.BOARD_SIZE*game.BOARD_SIZE) {
            std::cout << "Run out of turns, no win?" << std::endl;
            game.print_segments();
            break;
        }
    }
}

int test() {
    Hex::Game game(true);
    bool won;
    game.take_turn("D6",&won);
    game.take_turn("E6",&won);
    game.take_turn("F6",&won);
    game.take_turn("G6",&won);
    game.printhex();
    game.print_segments();
    game.take_turn("F5",&won);
    game.printhex();
    game.print_segments();
    game.take_turn("F4",&won);
    game.printhex();
    game.print_segments();
    game.take_turn("E5",&won);
    game.printhex();
    game.print_segments();
    game.take_turn("I5",&won);
    game.take_turn("I6",&won);
    game.take_turn("I7",&won);
    game.printhex();
    game.print_segments();
    game.take_turn("H6",&won);
    game.printhex();
    game.print_segments();
}

void test2() {
    Hex::Game game(true);
    bool won;
    game.take_turn("D4",&won);
    game.take_turn("D5",&won);
    game.printhex();
    game.print_segments();
    game.take_turn("C7",&won);
    game.take_turn("B8",&won);
    game.printhex();
    game.print_segments();
    game.take_turn("A9",&won);
    game.take_turn("K6",&won);
    game.printhex();
    game.print_segments();
    game.take_turn("E6",&won);
    game.printhex();
    game.print_segments();
    game.take_turn("F6",&won);
    game.printhex();
    game.print_segments();
    game.take_turn("G6",&won);
    game.take_turn("H6",&won);
    game.take_turn("I6",&won);
    game.printhex();
    game.print_segments();
    game.take_turn("J6",&won);
    game.printhex();
    game.print_segments();
    game.take_turn("D6",&won);
    game.printhex();
    game.print_segments();
    std::cout << won << std::endl;
}

int test2B () {
        Hex::Game game(true);
    bool won;
    game.take_turn("D4",&won);
    game.take_turn("D5",&won);
    game.printhex();
    game.print_segments();
    game.take_turn("C7",&won);
    game.take_turn("B8",&won);
    game.printhex();
    game.print_segments();
    game.take_turn("E6",&won);
    game.take_turn("F6",&won);
    game.take_turn("G6",&won);
    game.take_turn("H6",&won);
    game.take_turn("I6",&won);
    game.take_turn("J6",&won);
    game.take_turn("K6",&won);
    game.printhex();
    game.print_segments();
    game.take_turn("D6",&won);
    game.printhex();
    game.print_segments();
    std::cout << won << std::endl;
}

int main () {
    Hex::Game game(false);
    //bluewin(game);
    random_strategy(game);
    //test();
    //test2();
    //test2B();


    game.free_segments();
    return 0;

    char input[50];
    bool won;
    bool valid_move = true;
    while (1) {
        do {
            if (!valid_move) {
                std::cout << "Invalid move!" << std::endl;
            }
            std::cout << "Turn: ";
            std::cin >> input;
        }
        while (!(valid_move = game.take_turn(input, &won)));

        game.printhex();
        game.print_segments();
        if (won) {
            std::cout << "Won" << std::endl;
            break;
        }
    }
    game.free_segments();
    return 0;
}