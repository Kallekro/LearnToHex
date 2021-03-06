#ifndef STRATEGIES_H
#define STRATEGIES_H

#include "Hex.hpp"

#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConvolutionalModel.h>//single convolutional layer
#include <shark/Models/ConcatenatedModel.h>//for stacking layers, provides operator>>

using namespace shark;

namespace Hex {

/* Neural network strategies */

/***********************\
 * TD Network Strategy *
\***********************/
class TDNetworkStrategy : public Strategy {
private:
    LinearModel<RealVector, RectifierNeuron> m_inLayer;
    LinearModel<RealVector, RectifierNeuron> m_hiddenLayer;
    LinearModel<RealVector, LogisticNeuron> m_outLayer;
    ConcatenatedModel<RealVector> m_moveNet;

    // define input and output dimensions of network
    int inputDim = Hex::BOARD_SIZE * Hex::BOARD_SIZE;
    int outputDim = 1;
    // Define shape of hidden layer
    int hiddenIn = 80;
    int hiddenOut = 40;

    unsigned m_color;
    double m_epsilon = 0.1;

public:
	TDNetworkStrategy(){
        m_inLayer.setStructure(inputDim, hiddenIn);
        m_hiddenLayer.setStructure(hiddenIn, hiddenOut );
        m_outLayer.setStructure(hiddenOut , 1);
        m_moveNet = m_inLayer >> m_hiddenLayer >> m_outLayer;
    }

    void createInput( shark::blas::matrix<Tile>const& field, unsigned int activePlayer, RealVector& inputs) {
        inputs.clear();
        // encode board so active player's tiles are 1.0, opponent players tiles are -1.0 and empty tiles are 0.0
        for (int i=0; i < BOARD_SIZE; i++) {
            for (int j=0; j< BOARD_SIZE; j++) {
                if(field(i,j).tileState == activePlayer){       // Channel where players own tiles are
                    //inputs(3*(i*Hex::BOARD_SIZE+j)) = 1.0;
                    inputs(i*Hex::BOARD_SIZE+j) = 1.0;
                }
                else if(field(i,j).tileState != Hex::Empty ) {  // Channel where other players tiles are
                    //inputs(3*(i*Hex::BOARD_SIZE+j)+1) = 1.0;
                    inputs(i*Hex::BOARD_SIZE+j) = -1.0;
                }
            }
        }
    }

    // takes encoded inputs and evaluates model
    double evaluateNetwork(RealVector inputs) {
        RealVector outputs;
        m_moveNet.eval(inputs, outputs);
        return outputs[0];
    }

    // choose a move given possible move_values
    std::pair<double, int> chooseMove(std::vector<std::pair<double, int>> move_values, unsigned activeplayer, RealVector feasible_moves, bool epsilon_greedy) {
        std::pair<double, int> chosen_move( std::numeric_limits<double>::max() * (activeplayer==Blue ? -1 : -1) , -1 );
        double u = shark::random::uni(shark::random::globalRng(), 0.0, 1.0);
        if (epsilon_greedy && u < m_epsilon) {
            // if epsilon greedy we pick a random empty tile
            int ri = shark::random::uni(shark::random::globalRng(), 0, sum(feasible_moves)-1);
            int move_val_idx = 0;
            for (int i=0; i < feasible_moves.size(); i++) {
                if (feasible_moves(i) == 1) {
                    if (ri == 0) {
                        chosen_move = move_values[move_val_idx];
                        break;
                    }
                    ri--;
                    move_val_idx++;
                }
            }
        } else {
            // if not epsilon greedy, we pick the move with the lowest associated value
            for(int i=0; i < move_values.size(); i++) {
                if (move_values[i].first >= chosen_move.first ) {
                    chosen_move = move_values[i];
                }
            }
        }
        if (chosen_move.second == -1) {
            throw std::runtime_error("Chosen move is -1");
        }
        chosen_move.first = 1 - chosen_move.first;
        return chosen_move;
    }

    // returns a copy of game field
    shark::blas::matrix<Tile> getFieldCopy(shark::blas::matrix<Tile> field) {
        shark::blas::matrix<Tile> fieldCopy(BOARD_SIZE, BOARD_SIZE);
        for (int i=0; i < BOARD_SIZE; i++) {
            for (int j=0; j < BOARD_SIZE; j++) {
                fieldCopy(i,j) = Tile();
                fieldCopy(i,j).tileState = field(i,j).tileState;
            }
        }
        return fieldCopy;
    }

    // calculate all move values (a value for each feasible move)
    std::vector<std::pair<double, int>> getMoveValues(shark::blas::matrix<Tile>& fieldCopy, unsigned activePlayer, RealVector feasible_moves) {
        std::vector<std::pair<double, int>> move_values;
        int inputIdx=0;
        RealVector input((BOARD_SIZE*BOARD_SIZE), 0.0);

        double max;
        for (int i=0; i<feasible_moves.size(); i++) {
            if (feasible_moves(i) == 1) {
                int x = i % BOARD_SIZE;
                int y = i / BOARD_SIZE;
                fieldCopy(x,y).tileState = (TileState)activePlayer;
                fieldCopy = rotateField(fieldCopy, activePlayer == Red);
                createInput(fieldCopy, activePlayer == Blue ? Red : Blue, input);
                double value = this->evaluateNetwork(input);
                std::pair<double, int> valPair = std::pair<double, int>(value, i);
                move_values.push_back(valPair);
                fieldCopy(x,y).tileState = Empty;
            }
        }
        return move_values;
    }

    // choose an action given a board and feasible moves
    std::pair<double, int> getChosenMove(Game game, bool epsilon_greedy) {
        shark::blas::matrix<Tile> fieldCopy;
        unsigned activePlayer = game.ActivePlayer();
        if (activePlayer == Hex::Red) {
            fieldCopy = rotateField(game.getGameBoard(), false); // if player 2, rotate view counter-clockwise
        } else {
            fieldCopy = getFieldCopy(game.getGameBoard());
        }
        RealVector feasibleMoves = game.getFeasibleMoves(fieldCopy);
        std::vector<std::pair<double, int>> move_values = getMoveValues(fieldCopy, activePlayer, feasibleMoves);
        std::pair<double, int> chosen_move = chooseMove(move_values, activePlayer, feasibleMoves, epsilon_greedy);

        if (activePlayer == Hex::Red) {
            chosen_move.second = flipToOriginalRotatedIndex(chosen_move.second);
        }
        return chosen_move;
    }

    ConcatenatedModel<RealVector> GetMoveModel() override {
        return m_moveNet;
    };

    void setColor(unsigned color) {
        m_color = color;
    }

	std::size_t numParameters() const{
		return m_moveNet.numberOfParameters();
	}

    void setParameters(shark::RealVector const& parameters){
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

    RealVector getMoveAction(blas::matrix<Tile>const& field) override {}

    int type () override {
        return 2;
    }
};

/***************************\
 * CSA-ES Network Strategy *
\***************************/
class CSANetworkStrategy: public Hex::Strategy{
private:
	LinearModel<RealVector, RectifierNeuron> m_inLayer;
	LinearModel<RealVector, RectifierNeuron> m_hiddenLayer1;
	LinearModel<RealVector, RectifierNeuron> m_moveOut;

	ConcatenatedModel<RealVector> m_moveNet;

    // define input and output dimensions of network
    int inputDim = Hex::BOARD_SIZE * Hex::BOARD_SIZE;
    int outputDim = Hex::BOARD_SIZE*Hex::BOARD_SIZE;
    // Define shape of hidden layer
    int hiddenIn = 80;
    int hiddenOut = 40;

    unsigned m_color;

public:
	CSANetworkStrategy(){
		m_inLayer.setStructure(inputDim, hiddenIn );
		m_hiddenLayer1.setStructure(m_inLayer.outputShape(), hiddenOut);
		m_moveOut.setStructure(m_hiddenLayer1.outputShape(), outputDim);

        m_moveNet = m_inLayer >> m_hiddenLayer1 >> m_moveOut;
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
		RealVector inputs((Hex::BOARD_SIZE*Hex::BOARD_SIZE),0.0);
		for(int i = 0; i < Hex::BOARD_SIZE; i++){
			for(int j = 0; j < Hex::BOARD_SIZE; j++){
				if(field(i,j).tileState == m_color){ // Channel where players own tiles are 1
                    inputs((j*Hex::BOARD_SIZE+i)) = 1.0;
                } else if (field(i,j).tileState != Hex::Empty) {
                    inputs((j*Hex::BOARD_SIZE+i)) = -1.0;
                }
		    }
        }

		//Get raw response for everything
		RealVector response = m_moveNet(inputs);
		return response;
	}

	std::size_t numParameters() const override{
		return m_moveNet.numberOfParameters();
	}

	void setParameters(shark::RealVector const& parameters) override{
		auto p1 = subrange(parameters, 0, m_moveNet.numberOfParameters());
		m_moveNet.setParameterVector(p1);
	}

    ConcatenatedModel<RealVector> GetMoveModel() override {
        return m_moveNet;
    };

    int type () override {
        return 1;
    }
};

// Strategies for testing

/*******************\
 * Random Strategy *
\*******************/
class RandomStrategy : public Strategy {
public:
    shark::RealVector getMoveAction(shark::blas::matrix<Tile>const&) override{
        return shark::RealVector(BOARD_SIZE * BOARD_SIZE, 1.0);
    }

    std::size_t numParameters() const override{ return 1; }
    void setParameters(shark::RealVector const&) override{}
    ConcatenatedModel<RealVector> GetMoveModel() override{};

    int type () override {
        return 4;
    }
};

/******************\
 * Human Strategy *
\******************/
class HumanStrategy : public Strategy {
private:
    bool validInput(std::string inp, std::pair<unsigned,unsigned>* pos) {
        if (inp.length() == 0) { return false; }
        if (!((inp[0] >= 'a' && inp[0] <= ('a' + BOARD_SIZE)) || (inp[0] >= 'A' && inp[0] <= ('A' + BOARD_SIZE)))) {
            return false;
        }
        unsigned num;
        try {
            num = std::stoi(inp.substr(1, inp.length()-1));
        } catch (std::invalid_argument& e) {
            return false;
        }
        if (num <= 0 || num > BOARD_SIZE) { return false; }
        (*pos).first = num - 1;
        (*pos).second = toupper(inp[0]) - 65;
        return true;
    }
    bool m_forpython;
public:
    HumanStrategy(bool for_python) : m_forpython(for_python) {}

    shark::RealVector getMoveAction(shark::blas::matrix<Tile>const& field) override {
        if (m_forpython) {
            std::cout << "__TURN__" << std::endl;
        } else {
            std::cout << "Your turn." << std::endl;
        }
        std::string playerInput = "";
        std::pair<unsigned,unsigned> pos;
        bool validPos = false;
        while (!validPos) {
            while (!validInput(playerInput, &pos)) {
                if (playerInput.length() > 0) {
                    if (m_forpython) {
                        std::cout << "__INVALID_INPUT__" << std::endl;
                    } else {
                        std::cout << "Invalid input: " << playerInput << ". Please try again." << std::endl;
                    }
                }
                std::getline(std::cin, playerInput);
            }
            if (field(pos.first, pos.second).tileState == Empty) {
                validPos = true;
            } else {
                if (m_forpython) {
                    std::cout << "__NON_EMPTY__" << std::endl;
                } else {
                    std::cout << "Please select an empty tile." << std::endl;
                }
                playerInput = "";
            }
        }
        shark::RealVector movefield(BOARD_SIZE * BOARD_SIZE, -std::numeric_limits<double>::max());
        movefield(pos.first * BOARD_SIZE + pos.second) = 1.0;

        return movefield;
    }

    std::size_t numParameters() const override{ return 1; }
    void setParameters(shark::RealVector const&) override{}
    ConcatenatedModel<RealVector> GetMoveModel() override{};

    int type () override {
        return 3;
    }
};


}
#endif