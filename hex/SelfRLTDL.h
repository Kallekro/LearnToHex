#ifndef SelfRLTDL_H
#define SelfRLTDL_H
#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>

using namespace shark;

class SelfRLTDL : public AbstractSingleObjectiveOptimizer<RealVector >{
public:
	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	void init( ObjectiveFunctionType const& function, SearchPointType const& p) {
		SIZE_CHECK(p.size() == function.numberOfVariables());
		checkFeatures(function);
        m_searchPoint = p;
		//std::vector<RealVector> points(1,p);
		//std::vector<double> functionValues(1,0.0);
//
		//std::size_t lambda = SelfRLCMA::suggestLambda( p.size() );
		//doInit(
		//	points,
		//	functionValues,
		//	lambda,
		//	3.0/std::sqrt(double(p.size()))
		//);
	}


	void step(ObjectiveFunctionType const& function) {
        double res = function(m_searchPoint | m_searchPoint);

    }

private:
    SearchPointType m_searchPoint;
};


#endif