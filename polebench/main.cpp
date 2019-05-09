
#include "SelfRLCMA.h"
#include <shark/ObjectiveFunctions/Benchmarks/PoleSimulators/SinglePole.h>

#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConvolutionalModel.h>//single convolutional layer
#include <shark/Models/ConcatenatedModel.h>//for stacking layers, provides operator>>

using namespace shark;

//typedef boost::shared_ptr<SingleObjectiveFunction> Function;


class Strategy{
    public:
        virtual double getMoveAction(RealVector &v) = 0;
        virtual std::size_t numParameters() const = 0;
        virtual void setParameters(shark::RealVector const& parameters) = 0;
};

class NetworkStrategy: public Strategy{
private:
	LinearModel<RealVector, TanhNeuron> m_inLayer;
	LinearModel<RealVector, TanhNeuron> m_hiddenLayer;
	LinearModel<RealVector> m_outLayer;

	ConcatenatedModel<RealVector> m_moveNet;

    unsigned m_color;

public:
	NetworkStrategy(){
		m_inLayer.setStructure({4, 1, 1}, 10);
		m_hiddenLayer.setStructure(m_inLayer.outputShape(), 10);
		m_outLayer.setStructure(m_hiddenLayer.outputShape(), 1);
		m_moveNet = m_inLayer >> m_hiddenLayer >> m_outLayer;
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
	}

};

class SinglePoleTest : public SingleObjectiveFunction {
public:
	SinglePoleTest(std::size_t numberOfVariables)
    : m_numberOfVariables(numberOfVariables) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= IS_NOISY;
	}

	std::string name() const
	{ return "SinglePoleTest"; }

	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}

	bool hasScalableDimensionality()const{
		return true;
	}

	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_numberOfVariables = numberOfVariables;
	}

	SearchPointType proposeStartingPoint() const {
		return blas::normal(random::globalRng(), numberOfVariables(), 0.0, 1.0, shark::blas::cpu_tag());
	}

	double eval(SearchPointType const& x) const {
		SIZE_CHECK(x.size() == 2*numberOfVariables());
		m_evaluationCounter++;

		NetworkStrategy strat1;
		NetworkStrategy strat2;
		SinglePole singlePole1 = SinglePole(true);
		singlePole1.init();
		SinglePole singlePole2 = SinglePole(true);
		singlePole2.init();

		auto x0 = subrange(x, 0, numberOfVariables());
		auto x1 = subrange(x, numberOfVariables(), x.size());

		strat1.setParameters(x0);
		strat2.setParameters(x1);

		double maxcount = 1000.0;

		double count1 = 0;
		shark::RealVector v(numberOfVariables(), 0.0);
		std::vector<double> angles;
		//std::cout << std::endl;
		while ((!singlePole1.failure() && count1 < maxcount)) {
			singlePole1.getState(v);
			//if (abs(v(2)) < 0.001) { break; }
			double output = strat1.getMoveAction(v);
			if (output < 0) {
				output = 0;
			} else if (output > 1) {
				output = 1;
			}
			singlePole1.move(output);
			count1++;
		}
		double count2 = 0;
		//std::cout << std::endl;
		while ((!singlePole2.failure() && count2 < maxcount)) {
			singlePole2.getState(v);
			double output = strat2.getMoveAction(v);
			if (output < 0) {
				output = 0;
			} else if (output > 1) {
				output = 1;
			}
			singlePole2.move(output);
			count2++;
		}
		//std::cout << count1 << " --vs-- " << count2 << std::endl;

		if (count1 > count2) {
			return 1;
		} else if (count1 < count2 || random::coinToss(shark::random::globalRng())) {
			return 0;
		} else {
			return 1;
		}

    }
private:
	std::size_t m_numberOfVariables;
};

int main() {
    shark::random::globalRng().seed(1338);
	unsigned numvars = 4;
    SinglePoleTest single_pole = SinglePoleTest(numvars);
    SelfRLCMA cma;

    std::size_t lambda = SelfRLCMA::suggestLambda(numvars);

	NetworkStrategy strat;

	auto runExperiment=[&](){
		SinglePole singlePole = SinglePole(true);
		singlePole.init();

		strat.setParameters(cma.generatePolicy());

		double count = 0;
		double maxcount = 100000.0;
		shark::RealVector v(singlePole.noVars(), 0.0);
		while (!singlePole.failure() && count <= maxcount) {
			singlePole.getState(v);
			double output = strat.getMoveAction(v);
			singlePole.move(output);
			count++;
		}
		std::cout << "Reached count " << count << std::endl;
	};

    cma.init(single_pole, blas::normal(random::globalRng(), numvars, 0.0, 1.0, shark::blas::cpu_tag()), lambda, 1.0);
    for (std::size_t t = 0; t != 50000; ++t) {
        cma.step(single_pole);
		//runExperiment();
		if (t % 100 == 0) {
			std::cout << "t " << t << std::endl;
			std::cout << "sigma " << cma.sigma() << std::endl;
			std::cout << "rate " << cma.rate() << std::endl;
			runExperiment();
		}

    }
}









