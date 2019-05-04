#ifndef SelfRLTDL_H
#define SelfRLTDL_H
#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>

using namespace shark;

class SelfRLTDL : public AbstractSingleObjectiveOptimizer<RealVector >{
public:
	std::string name() const {
		return "sRLTDL"; 
	}

	static std::size_t suggestLambda( std::size_t dimension ) {
		std::size_t lambda = std::size_t( 4. + ::floor( 3 *::log( static_cast<double>( dimension ) ) ) );
		return lambda + lambda % 2;
	}

	void read (InArchive & archive ) {}
	void write (OutArchive & archive ) const{}

	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	void init( ObjectiveFunctionType const& function, SearchPointType const& p) {
		SIZE_CHECK(p.size() == function.numberOfVariables());
		checkFeatures(function);
        m_searchPoint = p;
		m_numberOfVariables = p.size();
	}

	void step(ObjectiveFunctionType const& function) {
        double res = function(m_searchPoint | m_searchPoint);
		// build matrix of logged states --- HOW? 
		// use createState on the concattenated model 
    }

private:
	std::size_t m_numberOfVariables;
	std::size_t m_lambda;

    SearchPointType m_searchPoint;

};


#endif