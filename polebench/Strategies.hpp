#ifndef STRATS_HPP
#define STRATS_HPP
/*
    Here the different strategies can be defined. A strategy contains
    model that the strategy its using.
    A strategy also implements action-choosing with getMoveAction, that gives 
    that is based on the policy that the model implements.
*/

#include <shark/Models/LinearModel.h>//single dense layer

using namespace shark;

/// Base strategy
class Strategy{
    public:
        virtual double getMoveAction(RealVector &v) = 0;
        virtual std::size_t numParameters() const = 0;
        virtual void setParameters(shark::RealVector const& parameters) = 0;
};

/// Network strategy
class NetworkStrategy: public Strategy{
private:
	LinearModel<RealVector, TanhNeuron> m_inLayer;
	LinearModel<RealVector, TanhNeuron> m_outLayer;
	LinearModel<RealVector, TanhNeuron> m_moveNet;

    unsigned m_color;

public:
	NetworkStrategy(){
		m_inLayer.setStructure({4, 1, 1}, 20);
		m_outLayer.setStructure(m_inLayer.outputShape(), 1);

		m_moveNet.setStructure(m_inLayer.inputShape(), 1); 
	}

	double getMoveAction(RealVector &inputs) override{
		RealVector response = m_moveNet(inputs);
		double result = response[0];
		return result;
	}

	std::size_t numParameters() const override{
		return m_moveNet.numberOfParameters();
	}

	void setParameters(shark::RealVector const& parameters) override{
		auto p1 = subrange(parameters, 0, m_moveNet.numberOfParameters());
		m_moveNet.setParameterVector(p1);
	}};

#endif