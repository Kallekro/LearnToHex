
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
            //double u = shark::random::uni(shark::random::globalRng(), 0.0, 1.0);
            //if (u > 0.5) {
            //    game.FlipBoard();
            //}
        }
		double r = game.getRank(0);
        return r;
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
		m_moveLayer2.setStructure(m_moveLayer.outputShape(),{10,4,4});
		//m_moveLayer3.setStructure(m_moveLayer2.outputShape(),{20,4,4});
		m_moveOut.setStructure(m_moveLayer2.outputShape(), {1,4,4});
		m_moveNet = m_moveLayer >> m_moveLayer2 >> m_moveOut;
	}
    void setColor(unsigned color) {
        m_color = color;
    }

	shark::RealVector getMoveAction(shark::blas::matrix<Hex::Tile>const& field) override{
		//find player position and prepare network position
		RealVector inputs((Hex::BOARD_SIZE*Hex::BOARD_SIZE*4),0.0);
		for(int i = 0; i != Hex::BOARD_SIZE; ++i){
			for(int j = 0; j != Hex::BOARD_SIZE; ++j){
                if (m_color == Hex::Red) {
                    int tmp = i;
                    i = j;
                    j = tmp;
                }
				if(field(i,j).tileState == m_color){ // Channel where players own tiles are 1
					inputs(4*(i*Hex::BOARD_SIZE+j)) = 1.0;
				}
				else if(field(i,j).tileState != Hex::Empty){ // Channel where other players tiles are 1
					inputs(4*(i*Hex::BOARD_SIZE+j)+1) = 1.0;
				}
				else if (  (m_color == Hex::Blue && (i == 0 || i == Hex::BOARD_SIZE-1))
                        || (m_color == Hex::Red  && (j == 0 || j == Hex::BOARD_SIZE-1))) {
					inputs(4*(i*Hex::BOARD_SIZE+j)+2) = 1.0;
				}
                else if (  (m_color == Hex::Red  && (i == 0 || i == Hex::BOARD_SIZE-1))
                        || (m_color == Hex::Blue && (j == 0 || j == Hex::BOARD_SIZE-1))) {
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
#if 0
    float wins_vs_random = 0;
    float games_vs_random_played = 0;
    std::deque<float> last_wins;
	for (std::size_t t = 0; t != 50000; ++t){
		//if(t % 1000 == 0)
        //    playGame();
        if(t % 10 == 0) {
            game.reset();
            std::cout << game.asciiState() << std::endl;
            while (game.takeTurn({&player1, &random_player})) {
                std::cout << game.asciiState() << std::endl;
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
    Hex::RandomStrategy random_player2;
    float wins_vs_random = 0;
    float games_vs_random_played = 0;
	for (std::size_t t = 0; t != 50000; ++t){
        game.reset();
        while (game.takeTurn({&random_player, &random_player2})) { }
        games_vs_random_played++;
        if (game.getRank(Hex::Blue)) {
            wins_vs_random++;
        }
        if (t%10000 == 0) {
            std::cout << games_vs_random_played << " games played. blue winrate: " << wins_vs_random / games_vs_random_played << std::endl;
        }
	}
    std::cout << "done." << std::endl;
    std::cout << "blue winrate: " << wins_vs_random / games_vs_random_played << std::endl;

#endif


}
