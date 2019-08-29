#include <boost/numeric/odeint.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

using namespace boost::numeric::odeint;

using state_t = std::vector<double>;

int mod_subtract(int a, int b, int bound){
	//computes a - b mod bound
	int res = a - b;
	if(res < 0){
		while(res < 0){
			res += bound;
		}
	}
	else if(res >= bound){
		res -= bound;
	}
	return res;
}

void time_differentiate(const state_t& x, state_t& dxdt, const double /* type param */)
{
	auto J = x.size();

	//precondition: x and dx/dt have the same length
	assert(J == dxdt.size());
	
	int bound = static_cast<long long>(J);

	double F = 8;

	for(int i = 0; i < bound; i++){
		dxdt[i] = ( x[(i+1) % bound] - x[mod_subtract(i, 2, bound)] ) * x[mod_subtract(i,1,bound)] - x[i] + F;
	}
}

void apply_initial_conditions(state_t& x){
	//apply initial conditions from the HW
	for(auto& val : x){
		val = 0;
	}
	x[1] = 1.0;
}

struct state_storage
{
	std::vector<state_t>& m_states;

	std::vector<double>& m_times;

	state_storage(std::vector<state_t>& states, std::vector<double>& times) :
																		m_states(states),
																		m_times(times) {};

	void operator()(const state_t& x, double t){
		m_states.push_back(x);
		m_times.push_back(t);
	}

};

int main(int argc, char** argv){

	state_t x(10);

	std::vector<state_t> x_stored;
	std::vector<double> times;

	apply_initial_conditions(x);

	double time_length = 10.0;

	double init_step_size = 0.05;

	std::cout << "Integrating ODE over 10 second total time length.\n";
	
	//solve with runge_kutta54_cash_karp stepper
	size_t steps = integrate(time_differentiate, x, 0.0,
						   	 time_length, init_step_size, 
							 state_storage(x_stored, times) );

	std::cout << "Solved ODE in " << steps << " timesteps.\n";

	std::ofstream filewriter;
	filewriter.open("hw1.csv");
	for(const auto& x : x_stored){
		for(const auto& val : x){
			filewriter << val << ", ";
		}
		filewriter << "\n";
	}

	filewriter.close();
	
	return 0;
}

	
	

