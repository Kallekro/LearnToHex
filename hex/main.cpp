
#include "SelfRLCMA.h"

#include "Hex.hpp"

//benchmark functions
#include <shark/ObjectiveFunctions/Benchmarks/Sphere.h>
#include <vector>


#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConvolutionalModel.h>//single convolutional layer
#include <shark/Models/ConcatenatedModel.h>//for stacking layers, provides operator>>

int main () {

    Hex::Game game;
    game.printhex();
    game.take_turn("A5");
    game.take_turn("E5");
    game.printhex();

    return 0;
}