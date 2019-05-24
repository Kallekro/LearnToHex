#include "Hex.hpp"

#include <vector>


#include <shark/Core/Shark.h>
#include <shark/Data/BatchInterface.h>
#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConvolutionalModel.h>//single convolutional layer
#include <shark/Models/ConcatenatedModel.h>//for stacking layers, provides operator>>

#include <stdlib.h>
#include <random>
#include <unistd.h>
#include <fstream>

using namespace shark;

class NetworkStrategy: public Hex::Strategy{
private:
    Conv2DModel<RealVector, FastSigmoidNeuron> m_inLayer;
    LinearModel<RealVector, FastSigmoidNeuron> m_hiddenLayer;
    LinearModel<RealVector, FastSigmoidNeuron> m_outLayer;

    size_t m_hidden1 = 40;

    ConcatenatedModel<RealVector> m_moveNet;

    unsigned m_color;

public:
	NetworkStrategy(){
		m_inLayer.setStructure({Hex::BOARD_SIZE, Hex::BOARD_SIZE, 2}, {20, 3, 3});
        m_hiddenLayer.setStructure(m_inLayer.outputShape(), m_hidden1);
        m_outLayer.setStructure(m_hiddenLayer.outputShape(), 1);
        m_moveNet = m_inLayer >> m_hiddenLayer >> m_outLayer;

        //m_moveNet.features |= 1; // HAS_FIRST_PARAMETER_DERIVATIVE
    }

    void createInput( shark::blas::matrix<Hex::Tile>const& field, unsigned int activePlayer, RealVector& inputs) {
        for (int i=0; i < Hex::BOARD_SIZE; i++) {
            for (int j=0; j< Hex::BOARD_SIZE; j++) {
                if(field(i,j).tileState == activePlayer){ // Channel where players own tiles are 1
                    if (activePlayer == Hex::Red ) {
                        inputs(2*(j*Hex::BOARD_SIZE+i)) = 1.0;
                    } else{
                        inputs(2*(i*Hex::BOARD_SIZE+j)) = 1.0;
                    }
                }
                else if(field(i,j).tileState != Hex::Empty ) { // Channel where other players tiles are 1
                    if (activePlayer == Hex::Red) {
                        inputs(2*(j*Hex::BOARD_SIZE+i)+1) = 1.0;
                    } else {
                        inputs(2*(i*Hex::BOARD_SIZE+j)+1) = 1.0;
                    }
                }
            }
        }
    }

    double evaluateNetwork(RealVector inputs) {
        RealVector outputs;
        m_moveNet.eval(inputs, outputs);
        return outputs(0);
    }

    ConcatenatedModel<RealVector> getModel() {
        return m_moveNet;
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


		return RealVector(2,0.0);
	}

	std::size_t numParameters() const override{
		return m_moveNet.numberOfParameters();
	}

    void setParameters(shark::RealVector const& parameters) override{
		auto p1 = subrange(parameters, 0, m_moveNet.numberOfParameters());
		m_moveNet.setParameterVector(p1);
	}
    void weightedParameterDerivative(RealMatrix input,
									 RealMatrix output,
									 RealMatrix weights,
									 State state,
									 RealVector derivative)
	{
		m_moveNet.weightedParameterDerivative(input, output, weights, state, derivative);
	}

    boost::shared_ptr<State> createState() {
        return m_moveNet.createState();
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

int main() {

    shark::random::globalRng().seed(time(NULL));
    double eps = 0.05;

    int num_episodes = 10000;
    double rate = 0.1;

    Hex::Game game(false);

    NetworkStrategy strat1;

    RealVector weights = shark::blas::normal(random::globalRng(), strat1.numParameters(), 0.0, 1.0/strat1.numParameters(), shark::blas::cpu_tag());

    for (int episode=0; episode < num_episodes; episode++) {
        game.reset();
        bool won = false;
        strat1.setParameters(weights);
       // std::cout << weights[0] << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
        // for computing derivatives
        std::vector<RealVector> states;
        RealVector rewards;
        RealVector values;
        RealVector nextValues;

        int step_i = 0;

        // save state
        RealVector input(2*(Hex::BOARD_SIZE*Hex::BOARD_SIZE), 0.0);
        //strat1.createInput(game.getGameBoard(), game.ActivePlayer(), input);
        states.push_back(input);
        values.push_back(strat1.evaluateNetwork(input));
        rewards.push_back(0);

        while (!won) {
            //std::cout << step_i << ", ";
            shark::blas::matrix<Hex::Tile> field = game.getGameBoard();
            unsigned int activePlayer = game.ActivePlayer();

            // save value of state S
            strat1.createInput(game.getGameBoard(), game.ActivePlayer(), input);
            values.push_back(strat1.evaluateNetwork(input));

            RealVector feasible_moves = game.getFeasibleMoves();
            std::vector<std::pair<double, int>> move_values;

            shark::blas::matrix<Hex::Tile> fieldCopy(Hex::BOARD_SIZE, Hex::BOARD_SIZE);

            for (int i=0; i < Hex::BOARD_SIZE; i++) {
                for (int j=0; j < Hex::BOARD_SIZE; j++) {
                    fieldCopy(i,j) = Hex::Tile();
                    fieldCopy(i,j).tileState = field(i,j).tileState;
                }
            }

            //std::cout << feasible_moves << std::endl;

            int inputIdx=0;
            for (int i=0; i<feasible_moves.size(); i++) {
                int x = i % Hex::BOARD_SIZE;
                int y = i / Hex::BOARD_SIZE;

                if (feasible_moves(i) == 1) {
                    fieldCopy(x,y).tileState = (Hex::TileState)activePlayer;
                    strat1.createInput(game.getGameBoard(), game.ActivePlayer(), input);
                    double value;
                    if (game.ActivePlayer() == Hex::Blue ) {
                        value = strat1.evaluateNetwork(input);
                    } else {
                        value = 1 - strat1.evaluateNetwork(input);
                    }
                    //value = strat1.evaluateNetwork(input);
                    move_values.push_back(std::pair<double, int>(value, i));
                    fieldCopy(x,y).tileState = Hex::Empty;
                }
                inputIdx += 2;
            }

            //for (int i=0; i < move_values.size(); i++) {
            //    std::cout << i << ": " << move_values[i].first << ", " << move_values[i].second << std::endl;
            //}


            std::pair<double, int> chosen_move( std::numeric_limits<double>::max() * (game.ActivePlayer()==Hex::Blue ? -1 : 1) , -1);

            double u = shark::random::uni(shark::random::globalRng(), 0.0, 1.0);
            if (u < eps) {
                int ri = shark::random::uni(shark::random::globalRng(), 0, sum(feasible_moves)-1);
                for (int i=0; i < feasible_moves.size(); i++) {
                    if (feasible_moves(i)) {
                        if (ri == 0) {
                            chosen_move = std::pair<double, int>(1.0, i);
                            break;
                        }
                        ri--;
                    }
                }
            } else {
                for(int i=0; i < move_values.size(); i++) {
                    if(game.ActivePlayer() == Hex::Blue && move_values[i].first >= chosen_move.first ) {
                        chosen_move = move_values[i];
                    } else if (game.ActivePlayer() == Hex::Red && move_values[i].first <= chosen_move.first) {
                        chosen_move = move_values[i];
                        chosen_move.first = 1 - chosen_move.first;
                    }
                }
            }
            // take action
            try {
                if (chosen_move.second < 0 || chosen_move.second >= Hex::BOARD_SIZE*Hex::BOARD_SIZE) {
                    std::cout << "Chosen move " << chosen_move.second << " out of range." << std::endl;
                    for (int i=0; i < move_values.size(); i++) {
                        std::cout << move_values[i].first;
                    }
                    std::cout << std::endl;
                    exit(1);
                } else {
                    won = !game.takeTurn(chosen_move.second);
                }

            } catch (std::invalid_argument& e) {
                std::cout << std::endl;
                throw(e);
            }

            // save state
            strat1.createInput(game.getGameBoard(), game.ActivePlayer(), input);
            states.push_back(input);
            // save reward
            if(!won ) {
                rewards.push_back(0);
            } else {
                //rewards.push_back(1);
                if (game.ActivePlayer() == Hex::Red) {
                    rewards.push_back(1);
                } else {
                    rewards.push_back(0);
                }
            }
            // save value
            nextValues.push_back(chosen_move.first);

            step_i++;
        }
        nextValues.push_back(rewards[step_i]);

        //std::cout << std::endl;
        if (episode % 100 == 0) {
            std::cout << "Game " << episode << std::endl;
            std::cout << game.asciiState() << std::endl;
        }

        RealVector statePoint(2*(Hex::BOARD_SIZE*Hex::BOARD_SIZE));
        RealVector valuePoint(1);
        // batch of states
        Batch<RealVector>::type stateBatch = Batch<RealVector>::createBatch(statePoint, step_i+1);
        // batch of values/outputs/predictions
        Batch<RealVector>::type valueBatch = Batch<RealVector>::createBatch(valuePoint, step_i+1);

        //std::cout << "ASSERT " << (values.size() == states.size())  << std::endl;
        //std::cout << values.size() << ", " << states.size() << std::endl;
        for (int i=0; i < states.size(); i++) {
            if (states[i].size() != 2*(Hex::BOARD_SIZE*Hex::BOARD_SIZE)) {
                std::cout << "state " << i << std::endl;
            }
            getBatchElement(stateBatch, i) = states[i];
            getBatchElement(valueBatch, i)(0) = values(i);
            //std::cout << states[i] << ": " << values(i) << std::endl;
        }


        //std::cout << "R: " << rewards.size() << " V: " << values.size() << " NV: " << nextValues.size() << std::endl;
//
        //std::cout << "SIZE statebatch: " <<  batchSize(stateBatch) << std::endl;
       // std::cout << "SIZE coeff size2: " <<  strat1.getModel().outputShape().numElements() << std::endl;
//
        //std::cout << "Rewards " << rewards << std::endl;
        //std::cout << "Values " << values << std::endl;
        //std::cout << "Next values " << nextValues << std::endl;

        // computes td-errors
        RealMatrix coeffs(step_i+1, strat1.getModel().outputShape().numElements()); // b td_main.cpp:286
        column(coeffs, 0) = rewards + nextValues - values;
        //RealVector coeffVec = rewards + nextValues - values; // TODO: values[::-1]
        //column(coeffs, 0) = coeffVec;
        //for (int i=0; i<step_i; i++) {
        //    coeffs(i, 0) =  rewards(i) + nextValues(i) - values(i);
        //}



        //std::cout << "Coeffs " << coeffs << std::endl;

        // batch of
        boost::shared_ptr<State> state = strat1.createState();
        strat1.getModel().eval(stateBatch, valueBatch, *state); // compiles

        RealVector derivative;
        strat1.getModel().weightedParameterDerivative(stateBatch, valueBatch, coeffs, *state, derivative); // b td_main.cpp:271
        //std::cout << "States: " << stateMat << std::endl;
        //std::cout << "Rewards: " << rewards << std::endl;
        //std::cout << "Values: " << values << std::endl;
        //std::cout << "NValues: " << nextValues << std::endl;
        //std::cout << "Derivatives: " << derivative << std::endl;
        //std::cout << strat1.getModel().numberOfParameters() << std::endl;
        //std::cout << "WWW  " << weights << std::endl;
        weights += rate*derivative;
    }

    return 0;
}