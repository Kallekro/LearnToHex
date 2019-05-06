
#include "SelfRLCMA.h"
#include "SelfRLTDL.h"

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
#include <fstream>

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
        m_features |= HAS_FIRST_DERIVATIVE;
	}

    Game getGame() const {
        return m_game;
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

		//reward of player 1
		//importance weighted reward
		double logW = 0.0;
		while(game.takeTurn({&strategy0, &strategy1})){
			logW += game.logImportanceWeight({&strategy0, &strategy1});
            //double u = shark::random::uni(shark::random::globalRng(), 0.0, 1.0);
            //if (u > 0.5) {
            //    game.FlipBoard();
            //}
        }

		double r = game.getRank(1);
		double y = 2*r - 1;

		double rr = 1/(1+std::exp(y*logW));
        return r;
	}
};



class NetworkStrategy: public Hex::Strategy{
private:
	Conv2DModel<RealVector, TanhNeuron> m_moveLayer;
	Conv2DModel<RealVector, TanhNeuron> m_moveLayer2;
    Conv2DModel<RealVector, TanhNeuron> m_moveLayer3;
	LinearModel<RealVector> m_moveOut;

	ConcatenatedModel<RealVector> m_moveNet ;

    unsigned m_color;

public:
	NetworkStrategy(){
		m_moveLayer.setStructure({Hex::BOARD_SIZE,Hex::BOARD_SIZE, 2}, {10,3,3});
		m_moveLayer2.setStructure(m_moveLayer.outputShape(),{20,3,3});
		m_moveOut.setStructure(m_moveLayer2.outputShape(), Hex::BOARD_SIZE*Hex::BOARD_SIZE);
		m_moveNet = m_moveLayer >> m_moveLayer2 >> m_moveOut;
	}

    void save(OutArchive & archive) {
        m_moveNet.write(archive);
    }
    void load(InArchive & archive) {
        m_moveNet.read(archive);
    }

    void setColor(unsigned color) {
        m_color = color;
    }

	shark::RealVector getMoveAction(shark::blas::matrix<Hex::Tile>const& field) override{
		//find player position and prepare network position
		RealVector inputs((Hex::BOARD_SIZE*Hex::BOARD_SIZE*2),0.0);
		for(int i = 0; i < Hex::BOARD_SIZE; i++){
			for(int j = 0; j < Hex::BOARD_SIZE; j++){
				if(field(i,j).tileState == m_color){ // Channel where players own tiles are 1
                    if (m_color == Hex::Red ) {
					    inputs(2*(j*Hex::BOARD_SIZE+i)) = 1.0;
                    } else{
					    inputs(2*(i*Hex::BOARD_SIZE+j)) = 1.0;
                    }
                }
				else if(field(i,j).tileState == Hex::Empty ) { // Channel where other players tiles are 1
                    if (m_color == Hex::Red) {
                        inputs(2*(j*Hex::BOARD_SIZE+i)+1) = 1.0;
                    } else {
                        inputs(2*(i*Hex::BOARD_SIZE+j)+1) = 1.0;
                    }
				}

				//else if (i == 0 || i == Hex::BOARD_SIZE-1) {
                        ////(m_color == Hex::Blue && (i == 0 || i == Hex::BOARD_SIZE-1))
                        ////|| (m_color == Hex::Red  && (j == 0 || j == Hex::BOARD_SIZE-1))) {
					//inputs(4*(i*Hex::BOARD_SIZE+j)+2) = 1.0;
				//}
                //else if (j == 0 || j == Hex::BOARD_SIZE-1) {
                        ////(m_color == Hex::Red  && (i == 0 || i == Hex::BOARD_SIZE-1))
                        ////|| (m_color == Hex::Blue && (j == 0 || j == Hex::BOARD_SIZE-1))) {
					//inputs(4*(i*Hex::BOARD_SIZE+j)+3) = 1.0;
				//}
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

void loadStrategy(std::string model_path, NetworkStrategy& strag) {
    std::ifstream ifs(model_path);
    boost::archive::polymorphic_text_iarchive ia(ifs);
    strag.load(ia);
    ifs.close();
}

void saveStrategy(std::string model_path, NetworkStrategy& strag) { 
    std::ostringstream name;
    name << model_path << ".model" ;
    std::ofstream ofs(name.str());
    boost::archive::polymorphic_text_oarchive oa(ofs);
    strag.save(oa);
    ofs.close();
}

int main () {
    Hex::Game game(false);

    shark::random::globalRng().seed(1338);

    NetworkStrategy player1;
    NetworkStrategy player2;
    player1.setColor(0);
    player2.setColor(1);

    Hex::RandomStrategy random_player;

    SelfPlayTwoPlayer<Hex::Game, NetworkStrategy> objective(game, player1);
    SelfRLCMA cma;
    SelfRLTDL tdl;
    std::size_t d = objective.numberOfVariables();
    std::size_t lambda = SelfRLCMA::suggestLambda(d);

    cma.init(objective, objective.proposeStartingPoint(), lambda, 1.0);

    //std::ifstream ifs("hex11.model");
    //boost::archive::polymorphic_text_iarchive ia(ifs);
    //player1.load(ia);
    //ifs.close();

    tdl.init(objective, objective.proposeStartingPoint());

    auto playGame=[&]() {
        game.reset();
        std::cout << game.asciiState() << std::endl;
        while (game.takeTurn({&player1, &player2})) {
            std::cout << game.asciiState() << std::endl;
        }
        std::cout << game.asciiState() << std::endl;
        std::cout << "End of example game." << std::endl;
    };
#if 1 // training test
    float wins_vs_random = 0;
    float games_vs_random_played = 0;
    std::deque<float> last_wins;

    unsigned version = 0;

	for (std::size_t t = 0; t != 50000; ++t){
        //playGame();

        if (t% 50 == 0) {
           playGame();
        }

        if (t%10 == 0) {
            // save the model
            saveStrategy("best", player1);
            version++;
        }

        if(t % 10 == 0 ) {
            game.reset();

            player1.setParameters(cma.mean());
            //player2.setParameters(cma.generatePolicy());

           // std::cout << game.asciiState() << std::endl;
            while (game.takeTurn({&player1, &random_player})) {
             //   std::cout << game.asciiState() << std::endl;
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
            std::cout<<"Game " << t << "\nSigma: " << cma.sigma() << std::endl;
            std::cout<< "Learn: " << cma.rate() << std::endl;
        }

     
        auto log = game.getLog();

        //for (int i=0; i < log.size(); i++){
            //std::cout << log[i].moveAction << std::endl;
        //}
   //     std::cout << log.size() << std::endl;
		cma.step(objective);
		//tdl.step(objective);
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
#elif 0 // Test with human player strategy
#if BUILD_FOR_PYTHON
    std::cout << "__BOARD_SIZE__ " << Hex::BOARD_SIZE << std::endl;
#endif
    shark::random::globalRng().seed(std::chrono::system_clock::now().time_since_epoch().count());
    Hex::HumanStrategy human_player;
    std::cout << game.asciiState() << std::endl;
    while (game.takeTurn({&human_player, &player2})) {
        std::cout << game.asciiState() << std::endl;
    }
    std::cout << game.asciiState() << std::endl;
    if (game.getRank(0)) {
        std::cout << "You won!" << std::endl;
    } else {
        std::cout << "You lost." << std::endl;
    }
    return 0;
#elif 0 // Test for win percentage of random players
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
