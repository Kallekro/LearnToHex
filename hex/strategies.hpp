#ifndef STRATEGIES_H
#define STRATEGIES_H

#include "Hex.hpp"

#include <shark/Core/Shark.h>
#include <shark/Data/BatchInterface.h>
#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConvolutionalModel.h>//single convolutional layer
#include <shark/Models/ConcatenatedModel.h>//for stacking layers, provides operator>>

using namespace shark;

#define EPSILON 0.1

class TDNetworkStrategy{
private:
    LinearModel<RealVector, FastSigmoidNeuron> m_inLayer;
    LinearModel<RealVector, FastSigmoidNeuron> m_hiddenLayer;
    LinearModel<RealVector, FastSigmoidNeuron> m_outLayer;

    ConcatenatedModel<RealVector> m_moveNet;

    unsigned m_color;

public:
	TDNetworkStrategy(){
		//m_inLayer.setStructure({Hex::BOARD_SIZE, Hex::BOARD_SIZE, 2}, {20, 3, 3});
        m_inLayer.setStructure(Hex::BOARD_SIZE*Hex::BOARD_SIZE*2, 10);
        m_hiddenLayer.setStructure(m_inLayer.outputShape(), 20);
        m_outLayer.setStructure(m_hiddenLayer.outputShape(), 1);
        m_moveNet = m_inLayer >> m_hiddenLayer >> m_outLayer;

        //m_moveNet.features |= 1; // HAS_FIRST_PARAMETER_DERIVATIVE
    }

    void createInput( shark::blas::matrix<Hex::Tile>const& field, unsigned int activePlayer, RealVector& inputs) {
        for (int i=0; i < Hex::BOARD_SIZE; i++) {
            for (int j=0; j< Hex::BOARD_SIZE; j++) {
                if(field(i,j).tileState == activePlayer){ // Channel where players own tiles are 1
                    // FLIP
                    //if (activePlayer == Hex::Red ) {
                    //    inputs(2*(j*Hex::BOARD_SIZE+i)) = 1.0;
                    //} else{
                    //    inputs(2*(i*Hex::BOARD_SIZE+j)) = 1.0;
                    //}
                    inputs(2*(i*Hex::BOARD_SIZE+j)) = 1.0;
                }
                else if(field(i,j).tileState != Hex::Empty ) { // Channel where other players tiles are 1
                    // FLIP
                    //if (activePlayer == Hex::Red) {
                    //    inputs(2*(j*Hex::BOARD_SIZE+i)+1) = 1.0;
                    //} else {
                    //    inputs(2*(i*Hex::BOARD_SIZE+j)+1) = 1.0;
                    //}
                    inputs(2*(i*Hex::BOARD_SIZE+j)+1) = 1.0;
                }
            }
        }
    }

    double evaluateNetwork(RealVector inputs) {
        RealVector outputs;
        m_moveNet.eval(inputs, outputs);
        return outputs(0);
    }


    std::pair<double, int> chooseMove(std::vector<std::pair<double, int>> move_values,
                                    unsigned activeplayer, RealVector feasible_moves,
                                    blas::matrix<Hex::Tile> field) {
        std::pair<double, int> chosen_move( std::numeric_limits<double>::max() * (activeplayer==Hex::Blue ? -1 : 1) , -1);
        double u = shark::random::uni(shark::random::globalRng(), 0.0, 1.0);
        if (u < EPSILON) {
            int ri = shark::random::uni(shark::random::globalRng(), 0, sum(feasible_moves)-1);
            for (int i=0; i < feasible_moves.size(); i++) {
                if (feasible_moves(i)) {
                    if (ri == 0) {
                        blas::matrix<Hex::Tile> fieldCopy = getFieldCopy(field);

                        fieldCopy(i % Hex::BOARD_SIZE, i / Hex::BOARD_SIZE).tileState = (Hex::TileState)activeplayer;
                        RealVector input(2*(Hex::BOARD_SIZE*Hex::BOARD_SIZE));
                        createInput(fieldCopy, activeplayer, input);
                        double val = evaluateNetwork(input);
                        chosen_move = std::pair<double, int>(val, i);
                        break;
                    }
                    ri--;
                }
            }
        } else {
            for(int i=0; i < move_values.size(); i++) {
                //if(move_values[i].first >= chosen_move.first ) {
                //    chosen_move = move_values[i];
                //}

                //std::cout << move_values[i].first << ", " << move_values[i].second << std::endl;

                if(activeplayer == Hex::Blue && move_values[i].first >= chosen_move.first ) {
                    chosen_move = move_values[i];
                } else if (activeplayer == Hex::Red && move_values[i].first <= chosen_move.first) {
                    chosen_move = move_values[i];
                    chosen_move.first = chosen_move.first;
                }
            }
        }
        return chosen_move;
    }

    shark::blas::matrix<Hex::Tile> getFieldCopy(shark::blas::matrix<Hex::Tile> field) {
        shark::blas::matrix<Hex::Tile> fieldCopy(Hex::BOARD_SIZE, Hex::BOARD_SIZE);
        for (int i=0; i < Hex::BOARD_SIZE; i++) {
            for (int j=0; j < Hex::BOARD_SIZE; j++) {
                fieldCopy(i,j) = Hex::Tile();
                fieldCopy(i,j).tileState = field(i,j).tileState;
            }
        }
        return fieldCopy;
    }


    std::vector<std::pair<double, int>> getMoveValues(shark::blas::matrix<Hex::Tile> fieldCopy, unsigned activePlayer, RealVector feasible_moves) {
        std::vector<std::pair<double, int>> move_values;
        int inputIdx=0;
        RealVector input(2*(Hex::BOARD_SIZE*Hex::BOARD_SIZE), 0.0);
        for (int i=0; i<feasible_moves.size(); i++) {
            int x = i % Hex::BOARD_SIZE;
            int y = i / Hex::BOARD_SIZE;

            if (feasible_moves(i) == 1) {
                fieldCopy(x,y).tileState = (Hex::TileState)activePlayer;
                this->createInput(fieldCopy, activePlayer, input);
                double value;
                if (activePlayer == Hex::Blue ) {
                    value = this->evaluateNetwork(input);
                } else {
                    value = 1 - this->evaluateNetwork(input);
                }
                move_values.push_back(std::pair<double, int>(value, i));
                fieldCopy(x,y).tileState = Hex::Empty;
            }
        }
        return move_values;
    }

    std::pair<double, int> getMoveAction(RealVector feasibleMoves, shark::blas::matrix<Hex::Tile> field, unsigned activePlayer) {
        shark::blas::matrix<Hex::Tile> fieldCopy = getFieldCopy(field);
        std::vector<std::pair<double, int>> move_values = getMoveValues(fieldCopy, activePlayer, feasibleMoves);
        std::pair<double, int> chosen_move = chooseMove(move_values, activePlayer, feasibleMoves);
        return chosen_move;
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

#endif