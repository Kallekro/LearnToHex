
#include "SelfRLCMA.h"

#include "Hex.hpp"

//benchmark functions
#include <shark/ObjectiveFunctions/Benchmarks/Sphere.h>
#include <vector>

#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConvolutionalModel.h>//single convolutional layer
#include <shark/Models/ConcatenatedModel.h>//for stacking layers, provides operator>>

#include <stdlib.h>
#include <random>
#include <unistd.h>

using namespace shark;


template<class Game, class Strategy>
class SelfPlayTwoPlayer : public SingleObjectiveFunction {
private:
	Game m_game;
	Strategy m_baseStrategy;
public:
	SelfPlayTwoPlayer(Game const& game, Strategy const& strategy)
	:m_game(game), m_baseStrategy(strategy){
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= IS_NOISY;
	}

	std::size_t numberOfVariables()const{
		return m_baseStrategy.numParameters();
	}

	SearchPointType proposeStartingPoint() const {
		return blas::normal(random::globalRng(), numberOfVariables(), 0.0, 1.0/numberOfVariables(), shark::blas::cpu_tag());
	}

	double eval(SearchPointType const& x) const {
		SIZE_CHECK(x.size() == 2*numberOfVariables());
		m_evaluationCounter++;

		auto x0 = subrange(x,0, numberOfVariables());
		auto x1 = subrange(x, numberOfVariables(), x.size());

		thread_local Strategy strategy0;
		thread_local Strategy strategy1;
		thread_local Game game = m_game;

		strategy0.setParameters(x0);
		strategy1.setParameters(x1);

		//simulate
		game.reset();

		//importance weighted reward
		double logW = 0.0;
		while(game.takeTurn({&strategy0, &strategy1})){
			logW += game.logImportanceWeight({&strategy1, &strategy0});
            double u = shark::random::uni(shark::random::globalRng(), 0.0, 1.0);
            if (u > 0.5) {
                game.FlipBoard();
            }
        }
		double r = game.getRank(0);
		double y = 2*r - 1;
		return 1/(1+std::exp(y*logW));
	}
};



class NetworkStrategy: public Hex::Strategy{
private:
	Conv2DModel<RealVector, TanhNeuron> m_moveLayer;
	Conv2DModel<RealVector, TanhNeuron> m_moveLayer2;
    Conv2DModel<RealVector, TanhNeuron> m_moveLayer3;
	Conv2DModel<RealVector> m_moveOut;

	ConcatenatedModel<RealVector> m_moveNet ;

    unsigned m_color;

public:
	NetworkStrategy(){
		m_moveLayer.setStructure({Hex::BOARD_SIZE,Hex::BOARD_SIZE, 4},{10,4,4});
		m_moveLayer2.setStructure(m_moveLayer.outputShape(),{20,4,4});
		m_moveLayer3.setStructure(m_moveLayer2.outputShape(),{20,4,4});
		m_moveOut.setStructure(m_moveLayer3.outputShape(), {1,4,4});
		m_moveNet = m_moveLayer >> m_moveLayer2 >> m_moveLayer3 >> m_moveOut;
	}
    void setColor(unsigned color) {
        m_color = color;
    }

	shark::RealVector getMoveAction(shark::blas::matrix<Hex::Tile>const& field) override{
		//find player position and prepare network position
		RealVector inputs((Hex::BOARD_SIZE*Hex::BOARD_SIZE*4),0.0);
		for(int i = 0; i != Hex::BOARD_SIZE; ++i){
			for(int j = 0; j != Hex::BOARD_SIZE; ++j){
				if(field(i,j).tileState == m_color){ // Channel where players own tiles are 1
					inputs(4*(i*Hex::BOARD_SIZE+j)) = 1.0;
				}
				else if(field(i,j).tileState != Hex::Empty){ // Channel where other players tiles are 1
					inputs(4*(i*Hex::BOARD_SIZE+j)+1) = 1.0;
				}
				else if (( m_color == Hex::Blue && (i == 0 || i == Hex::BOARD_SIZE-1))
                        || m_color == Hex::Red  && (j == 0 || j == Hex::BOARD_SIZE-1)) {
					inputs(4*(i*Hex::BOARD_SIZE+j)+2) = 1.0;
				}
                else if (( m_color == Hex::Red  && (i == 0 || i == Hex::BOARD_SIZE-1))
                        || m_color == Hex::Blue && (j == 0 || j == Hex::BOARD_SIZE-1)) {
					inputs(4*(i*Hex::BOARD_SIZE+j)+3) = 1.0;
				}
			}
		}

		//Get raw response for everything
		RealVector response = m_moveNet(inputs);

        // hmm
        //response(Hex::BOARD_SIZE*Hex::BOARD_SIZE - 1) = 0.0;

		////return only the output at player position
		//RealVector output(121, 1.0);
		//for(std::size_t i = 0; i != 7; ++i){
			//output(i) = response(7*(y*7+x) + i);
		//}
		return response;

	}

	std::size_t numParameters() const override{
		return m_moveNet.numberOfParameters();
	}

	void setParameters(shark::RealVector const& parameters) override{
		auto p1 = subrange(parameters, 0, m_moveNet.numberOfParameters());
		m_moveNet.setParameterVector(p1);
	}

};

// TODO:
// * implement game.reset()
// * implement game.asciiState() (like printhex() but returns string)
int main () {
    Hex::Game game(false);

    shark::random::globalRng().seed(1338);

    NetworkStrategy player1;
    NetworkStrategy player2;
    player1.setColor(0);
    player2.setColor(1);

    //Hex::RandomStrategy player1;
    Hex::RandomStrategy random_player;

    SelfPlayTwoPlayer<Hex::Game, NetworkStrategy> objective(game, player1);
    SelfRLCMA cma;
    std::size_t d = objective.numberOfVariables();
    std::size_t lambda = SelfRLCMA::suggestLambda(d);
    cma.init(objective, objective.proposeStartingPoint(), lambda, 1.0);

    auto playGame=[&]() {
        player1.setParameters(cma.generatePolicy());
        player2.setParameters(cma.generatePolicy());
        game.reset();
        std::cout << game.asciiState() << std::endl;
        while (game.takeTurn({&player1, &player2})) {
            std::cout << game.asciiState() << std::endl;
        }
        std::cout << game.asciiState() << std::endl;
    };
#if 1
    float wins_vs_random = 0;
    float games_vs_random_played = 0;
    std::deque<float> last_wins;
	for (std::size_t t = 0; t != 50000; ++t){
		//if(t % 1000 == 0)
        //    playGame();
        if(t % 10 == 0) {
            game.reset();
            //std::cout << game.asciiState() << std::endl;
            while (game.takeTurn({&player1, &random_player})) {
                //std::cout << game.asciiState() << std::endl;
            }
            std::cout << game.asciiState() << std::endl;
            std::cout << "end of random game" << std::endl;
            games_vs_random_played++;
            if (game.getRank(Hex::Blue)) {
                std::cout << "Network strategy (blue) won!" << std::endl;
                wins_vs_random++;
                last_wins.push_back(1);
            } else {
                std::cout << "Random strategy (red) won!" << std::endl;
                last_wins.push_back(0);
            }
            std::cout << "blue winrate: " << wins_vs_random / games_vs_random_played << std::endl;
            if (last_wins.size() > 100) {
                last_wins.pop_front();
            }
            float sum = 0;
            for (int i=0; i < last_wins.size(); i++) {
                sum += last_wins[i];
            }
            std::cout << "blue winrate last " << last_wins.size() << " games: " << sum / last_wins.size() << std::endl;

            std::cout <<t<<" "<<cma.sigma()<<std::endl;
        }
        game.reset();
		cma.step(objective);
	}

	std::cout<<"optimization done. example game"<<std::endl;
	player1.setParameters(cma.generatePolicy());
	player2.setParameters(cma.generatePolicy());
	game.reset();
	std::cout<<game.asciiState()<<std::endl;
	while(game.takeTurn({&player1, &player2})){
		std::cout<<game.asciiState()<<std::endl;
	}

    return 0;
#else
    char buf[10];
    buf[0] = '\0';
    while (strcmp(buf, "q") != 0) {
        cma.step(objective);
        std::cout << "Enter anything to update step" << std::endl;
        std::cin >> buf;
    }

#endif


}
/*
void random_strategy (Hex::Game game) {

    std::random_device rd;
    std::mt19937 rng_engine(rd());
    std::uniform_int_distribution<int> dist(1,BOARD_SIZE);

    bool won;
    bool valid_move = true;
    std::string random_tile = "";

    char letters[BOARD_SIZE];

    for (char i = 65; i < 65+BOARD_SIZE; i++) {
        letters[i-65] = i;
    }

    int turn_count = 0;
    while (1) {
        do {
            auto randint1 = dist(rng_engine);
            auto randint2 = dist(rng_engine);
            random_tile = letters[randint1-1] + std::to_string(randint2);
        }
        while (!(valid_move = game.take_turn(random_tile, &won)));
        //game.printhex();
        if (won) {
            std::cout << game.ActivePlayer() << " LOST" << std::endl;
            break;
        }
        turn_count++;
        if (turn_count >= BOARD_SIZE * BOARD_SIZE) {
            std::cout << "Run out of turns, no win?" << std::endl;
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
    //game.print_segments();
    game.take_turn("F5",&won);
    game.printhex();
    //game.print_segments();
    game.take_turn("F4",&won);
    game.printhex();
    //game.print_segments();
    game.take_turn("E5",&won);
    game.printhex();
    //game.print_segments();
    game.take_turn("I5",&won);
    game.take_turn("I6",&won);
    game.take_turn("I7",&won);
    game.printhex();
    //game.print_segments();
    game.take_turn("H6",&won);
    game.printhex();
    //game.print_segments();
}

void test2() {
    Hex::Game game(true);
    bool won;
    game.take_turn("D4",&won);
    game.take_turn("D5",&won);
    game.printhex();
    //game.print_segments();
    game.take_turn("C7",&won);
    game.take_turn("B8",&won);
    game.printhex();
    //game.print_segments();
    game.take_turn("A9",&won);
    game.take_turn("K6",&won);
    game.printhex();
    //game.print_segments();
    game.take_turn("E6",&won);
    game.printhex();
    //game.print_segments();
    game.take_turn("F6",&won);
    game.printhex();
    //game.print_segments();
    game.take_turn("G6",&won);
    game.take_turn("H6",&won);
    game.take_turn("I6",&won);
    game.printhex();
    //game.print_segments();
    game.take_turn("J6",&won);
    game.printhex();
    //game.print_segments();
    game.take_turn("D6",&won);
    game.printhex();
    //game.print_segments();
    std::cout << won << std::endl;
}

int test2B () {
        Hex::Game game(true);
    bool won;
    game.take_turn("D4",&won);
    game.take_turn("D5",&won);
    game.printhex();
    //game.print_segments();
    game.take_turn("C7",&won);
    game.take_turn("B8",&won);
    game.printhex();
    //game.print_segments();
    game.take_turn("E6",&won);
    game.take_turn("F6",&won);
    game.take_turn("G6",&won);
    game.take_turn("H6",&won);
    game.take_turn("I6",&won);
    game.take_turn("J6",&won);
    game.take_turn("K6",&won);
    game.printhex();
    //game.print_segments();
    game.take_turn("D6",&won);
    game.printhex();
    //game.print_segments();
    std::cout << won << std::endl;
}

void negative_test() {
    //Hex::Game game(true);
    //game.take_turn("A2",)
}

void maximum_linesegments() {
    Hex::Game game(false);
    bool won;
    for (int j=1; j < 12; j+= 2) {
        for (char i=65; i < 65 + 11; i++) {
            char move[3];
            sprintf(move, "%c%d", i, j);
            game.take_turn(move, &won);
        }
    }
    game.printhex();
}

void minimum_linesegments() {
    Hex::Game game(false);
    bool won;
    game.take_turn("A1", &won);  game.take_turn("K11", &won);
    game.take_turn("A2", &won);  game.take_turn("J11", &won);
    game.take_turn("A3", &won);  game.take_turn("I11", &won);
    game.take_turn("A4", &won);  game.take_turn("H11", &won);
    game.take_turn("A5", &won);  game.take_turn("G11", &won);
    game.take_turn("A6", &won);  game.take_turn("F11", &won);
    game.take_turn("A7", &won);  game.take_turn("E11", &won);
    game.take_turn("A8", &won);  game.take_turn("D11", &won);
    game.take_turn("A9", &won);  game.take_turn("C11", &won);
    game.take_turn("A10", &won); game.take_turn("B11", &won);
    game.take_turn("A11", &won); //game.take_turn("K", &won);
    game.printhex();
}
*/