#include "enkf.hpp"
#include <Eigen/Core>
#include <vector>
#include <fstream>

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

Vec Henon_Map(Vec& x, double& a, double& b){
  auto xn = x[0];
  auto yn = x[1];

  x[0] = 1 - a * xn * xn + yn;
  x[1] = b * xn;

  return x;
}

double obs_err(){
   std::random_device rd{};
   std::mt19937 gen{rd()};
   std::normal_distribution<> dis{0.0, std::sqrt(0.02)};

   return dis(gen);
}

int main(int argc, char** argv){
  using namespace fastmath;

  std::function<Vec(Vec&, double&, double&)> fmap = Henon_Map;
  int ensemble_size = 31;
  auto initial_state = Vec::Zero(2);

  Mat H(1,2);

  H << 1.0, 0.0;

  Vec true_state(2);
  true_state << 0.0, 0.0;

  Vec ensemble_mean = Vec::Zero(ensemble_size);

  Mat ensemble_cov = 0.01 * Mat::Identity(2,2);

  Mat obs_err_cov = 0.02 * Mat::Identity(1,1);

  int n_assim_cycles = 100;

  double a = 1.4, b = 0.3;
  Mat state_mat(2, n_assim_cycles);
  for(auto i = 0; i < n_assim_cycles; ++i){
    true_state = Henon_Map(true_state, a, b);
    state_mat.col(i) = true_state;
  }

  std::vector<double> filter_err, etkf_err;

  auto enkf_runner = VectorEnKF<double, double>(fmap, H, initial_state,
						obs_err_cov, initial_state,
						ensemble_cov, ensemble_size);

  auto etkf_runner = VectorEnKF<double, double>(fmap, H, initial_state,
						obs_err_cov, initial_state,
						ensemble_cov, ensemble_size);

  Vec y(1);
  double yval;
  Vec analysis_mean;
  for(auto i = 0; i < n_assim_cycles; ++i){
    true_state = state_mat.col(i);
    yval = true_state[0] + obs_err();
    y[0] = yval;
    enkf_runner.filter(y, a, b);
    etkf_runner.filter(y, a, b);
    analysis_mean = enkf_runner.state();
    Vec etkf_mean = etkf_runner.state();
    Vec err = true_state - analysis_mean;
    filter_err.push_back(err.norm());
    err = true_state - etkf_mean;
    etkf_err.push_back(err.norm());
  }

  std::ofstream fwriter("hw7.csv"), twriter("hw7etkf.csv");
  for(const auto& e : filter_err){
    fwriter << e << '\n';
  }

  for(const auto& e : etkf_err){
    twriter << e << '\n';
  }

  return 0;
}
  


						
						
  
  
