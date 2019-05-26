#ifndef STRATEGIES_H
#define STRATEGIES_H

#include "Hex.hpp"

#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConvolutionalModel.h>//single convolutional layer
#include <shark/Models/ConcatenatedModel.h>//for stacking layers, provides operator>>

using namespace shark;

#define EPSILON 0.1

namespace Hex {

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
};

/* Neural network strategies */

/************************\
 * CMA Network Strategy *
\************************/
class CMANetworkStrategy: public Hex::Strategy{
private:
	Conv2DModel<RealVector, TanhNeuron> m_inLayer;
	LinearModel<RealVector, TanhNeuron> m_hiddenLayer1;
	LinearModel<RealVector> m_moveOut;

	ConcatenatedModel<RealVector> m_moveNet ;

    unsigned m_color;

public:
	CMANetworkStrategy(){
		m_inLayer.setStructure({Hex::BOARD_SIZE,Hex::BOARD_SIZE, 2}, {20, 3,1});
		m_hiddenLayer1.setStructure(m_inLayer.outputShape(), {18,Hex::BOARD_SIZE,Hex::BOARD_SIZE});
//        m_moveLayer3.setStructure(m_moveLayer2.outputShape(), {60,3,3});
		m_moveOut.setStructure(m_inLayer.outputShape(), Hex::BOARD_SIZE*Hex::BOARD_SIZE);
        m_moveNet = m_inLayer >> m_moveOut;
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
				else if(field(i,j).tileState != Hex::Empty ) { // Channel where other players tiles are 1
                    if (m_color == Hex::Red) {
                        inputs(2*(j*Hex::BOARD_SIZE+i)+1) = 1.0;
                    } else {
                        inputs(2*(i*Hex::BOARD_SIZE+j)+1) = 1.0;
                    }
				}

				//if ( (m_color == Hex::Blue && (i == 0 || i == Hex::BOARD_SIZE-1)) || (m_color == Hex::Red  && (j == 0 || j == Hex::BOARD_SIZE-1))) {
					//inputs(4*(i*Hex::BOARD_SIZE+j)+2) = 1.0;
				//}
                //if ((m_color == Hex::Red  && (i == 0 || i == Hex::BOARD_SIZE-1)) || (m_color == Hex::Blue && (j == 0 || j == Hex::BOARD_SIZE-1))) {
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

    ConcatenatedModel<RealVector> GetMoveModel() override {
        return m_moveNet;
    };

};

/***********************\
 * TD Network Strategy *
\***********************/
class TDNetworkStrategy : public Strategy {
private:
    LinearModel<RealVector, FastSigmoidNeuron> m_inLayer;
    LinearModel<RealVector, FastSigmoidNeuron> m_hiddenLayer;
    LinearModel<RealVector, FastSigmoidNeuron> m_outLayer;

    ConcatenatedModel<RealVector> m_moveNet;

    unsigned m_color;

public:
	TDNetworkStrategy(){
		//m_inLayer.setStructure({BOARD_SIZE, BOARD_SIZE, 2}, {20, 3, 3});
        m_inLayer.setStructure(Hex::BOARD_SIZE*Hex::BOARD_SIZE * 3, Hex::BOARD_SIZE*Hex::BOARD_SIZE);
        m_hiddenLayer.setStructure(m_inLayer.outputShape(), 40 );
        m_outLayer.setStructure(m_hiddenLayer.outputShape(), 1);
        m_moveNet = m_inLayer >> m_hiddenLayer >> m_outLayer;

        //m_moveNet.features |= 1; // HAS_FIRST_PARAMETER_DERIVATIVE
    }

    void createInput( shark::blas::matrix<Tile>const& field, unsigned int activePlayer, RealVector& inputs) {
        for (int i=0; i < BOARD_SIZE; i++) {
            for (int j=0; j< BOARD_SIZE; j++) {
                if(field(i,j).tileState == activePlayer){       // Channel where players own tiles(looks as blue) are 1
                    inputs(3*(i*Hex::BOARD_SIZE+j)) = 1.0;
                }
                else if(field(i,j).tileState != Hex::Empty ) {  // Channel where other(red) players tiles are 1
                    inputs(3*(i*Hex::BOARD_SIZE+j)+1) = 1.0;
                } else {                                        // Empty tile
                    inputs(3*(i*Hex::BOARD_SIZE+j)+2) = 1.0;
                }
            }
        }
    }

    blas::vector<Hex::Tile> reverseVector(blas::vector<Hex::Tile> vec) {
        blas::vector<Hex::Tile> vecOut(vec.size());
        for (int i=0; i < vec.size(); i++) {
            vecOut(vec.size() - i - 1) = vec(i);
        }
        return vecOut;
    }

    shark::blas::matrix<Hex::Tile> rotateField(shark::blas::matrix<Hex::Tile>const& field) {
        //int n = Hex::BOARD_SIZE;
        //int f = floor(n / 2);
        //int c = ceil(n / 2);
        //shark::blas::matrix<Hex::Tile> fieldCopy(n, n);
        ////RealVector input(3*(Hex::BOARD_SIZE*Hex::BOARD_SIZE));
        ////createInput(field, Hex::Red, input);
        ////std::cout <<"Before: " <<  input << std::endl;
        //for (int i=0; i < f; i++) {
        //    for (int j=0; j < c; j++) {
        //        Hex::TileState tmp = field(i,j).tileState;
        //        fieldCopy(i,j).tileState = field(j, n - 1 - i).tileState;
        //        fieldCopy(j, n - 1 - i).tileState = field(n - 1 - i, n - 1 - j ).tileState;
        //        fieldCopy(n - 1 - i, n - 1 - j).tileState = field(n - 1 - j, i).tileState;
        //        fieldCopy(n-1-j, i).tileState = tmp;
        //    }
        //}
        //input.clear();
        //createInput(fieldCopy, Hex::Red, input);
        //std::cout <<"After: " <<  input << std::endl;

        //RealVector input(3*(Hex::BOARD_SIZE*Hex::BOARD_SIZE));
        //createInput(field, Hex::Red, input);
        //std::cout <<"Before: " <<  input << std::endl;
        shark::blas::matrix<Hex::Tile> fieldCopy(BOARD_SIZE, BOARD_SIZE);
        for (int i=0; i < BOARD_SIZE; i++) {
            column(fieldCopy, i) = reverseVector(row(field, i));
            //for (int j=0; j < BOARD_SIZE; j++) {
            //    fieldCopy(BOARD_SIZE - i, BOARD_SIZE - 1 - i) = field(i, j);
            //}
        }
        //input.clear();
        //createInput(fieldCopy, Hex::Red, input);
        //std::cout <<"After: " <<  input << std::endl;
        return fieldCopy;
    }

    double evaluateNetwork(RealVector inputs) {
        RealVector outputs;
        m_moveNet.eval(inputs, outputs);
        return outputs(0);
    }


    std::pair<double, int> chooseMove(std::vector<std::pair<double, int>> move_values, unsigned activeplayer, RealVector feasible_moves, bool epsilon_greedy) {
        std::pair<double, int> chosen_move( std::numeric_limits<double>::max() * (activeplayer==Blue ? -1 : 1) , -1);
        double u = shark::random::uni(shark::random::globalRng(), 0.0, 1.0);
        if (epsilon_greedy && u < EPSILON) {
            int ri = shark::random::uni(shark::random::globalRng(), 0, sum(feasible_moves)-1);
            for (int i=0; i < feasible_moves.size(); i++) {
                if (feasible_moves(i)) {
                    if (ri == 0) {
                        chosen_move = std::pair<double, int>(move_values[i].first, i);
                        break;
                    }
                    ri--;
                }
            }
        } else {
            for(int i=0; i < move_values.size(); i++) {
                if(activeplayer == Blue && move_values[i].first >= chosen_move.first ) {
                    chosen_move = move_values[i];
                } else if (activeplayer == Red && move_values[i].first <= chosen_move.first) {
                    chosen_move = move_values[i];
                    chosen_move.first = chosen_move.first;
                }
            }
        }
        return chosen_move;
    }

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

    RealVector getMoveAction(blas::matrix<Tile>const& field) override {}

    std::vector<std::pair<double, int>> getMoveValues(shark::blas::matrix<Tile> fieldCopy, unsigned activePlayer, RealVector feasible_moves) {
        std::vector<std::pair<double, int>> move_values;
        int inputIdx=0;
        RealVector input(3*(BOARD_SIZE*BOARD_SIZE), 0.0);
        for (int i=0; i<feasible_moves.size(); i++) {
            int x = i % BOARD_SIZE;
            int y = i / BOARD_SIZE;

            if (feasible_moves(i) == 1) {
                fieldCopy(x,y).tileState = (TileState)activePlayer;
                this->createInput(fieldCopy, activePlayer, input);
                double value;
                if (activePlayer == Blue ) {
                    value = this->evaluateNetwork(input);
                } else {
                    value = this->evaluateNetwork(input);
                }
                move_values.push_back(std::pair<double, int>(value, i));
                fieldCopy(x,y).tileState = Empty;
            }
        }
        return move_values;
    }

    std::pair<double, int> getChosenMove(RealVector feasibleMoves, shark::blas::matrix<Tile> field, unsigned activePlayer, bool epsilon_greedy) {
        if (activePlayer == Hex::Red) {
            field = rotateField(field); // if player 2, rotate view
        }
        shark::blas::matrix<Tile> fieldCopy = getFieldCopy(field);
        std::vector<std::pair<double, int>> move_values = getMoveValues(fieldCopy, activePlayer, feasibleMoves);
        std::pair<double, int> chosen_move = chooseMove(move_values, activePlayer, feasibleMoves, epsilon_greedy);
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
};
}

#endif