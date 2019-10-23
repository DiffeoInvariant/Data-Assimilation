#ifndef ENKF_HPP
#define ENKF_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <vector>
#include <functional>
#include <utility>
#include <cassert>
#include <random>
#include <cmath>
#include <iostream>

namespace fastmath
{
  using Vec = Eigen::VectorXd;
  using Mat = Eigen::MatrixXd;

  template<class State_t, class Obs_t, class ObsOp_t, class... ForecastArgs>
  class EnKF
  {
  public:
    
    constexpr EnKF() {};

    //filter with one observation
    /* virtual void filter(const Obs_t& y);*/

    virtual Obs_t observe(const State_t& x);

    virtual State_t forecast(State_t& x, ForecastArgs&... args);

    virtual ~EnKF() {};
  };

  //scalar EnKF with linear observations
  template<class... ForecastArgs>
  class VectorEnKF : public EnKF<Vec, Vec, Mat, ForecastArgs...>
  {

 private:

    Mat  m_H;

    Vec m_state;

    Vec m_obs = m_H * m_state;

    std::function<Vec(Vec&, ForecastArgs&...)> m_forecast;

    Vec m_ensemble_mean;

    Mat m_ensemble_covariance;

    Mat m_obs_covariance;

    int m_ensemble_size;

    int m_dimension;

    Mat m_ensemble = Mat::Zero(m_dimension, m_ensemble_size);

    bool m_initialized_ensemble = false;

    Mat m_background_covariance = Mat::Identity(m_dimension, m_dimension);

    Mat m_ensemble_transform = gen_covariance_transform(m_ensemble_covariance);
  public:

    VectorEnKF(const std::function<Vec(Vec&, ForecastArgs&...)>& n_forecast,
			 const Mat& n_H,
			 const Vec& n_initial_state,
			 const Mat& n_observe_err_covariance,
			 const Vec& n_initial_ensemble_mean,
			 const Mat&   n_initial_ensemble_covariance,
			 const int    n_ensemble_size
			 ) : EnKF<Vec, Vec, Mat, ForecastArgs...>(),
			     m_H(n_H),
			     m_state(n_initial_state),
			     m_forecast(n_forecast),
			     m_ensemble_mean(n_initial_ensemble_mean),
			     m_ensemble_covariance(n_initial_ensemble_covariance),
			     m_obs_covariance(n_observe_err_covariance),
			     m_ensemble_size(n_ensemble_size),
			     m_dimension(static_cast<int>(n_initial_state.size()))
			     
    {};


    void set_state(const Vec& n_state) noexcept
    {
      m_state = n_state;
    }

    void set_ensemble_size(const int n_ensemble_size) noexcept
    {
      assert(n_ensemble_size > 0);
      m_ensemble_size = n_ensemble_size;
    }
     

    Vec state()
    {
      return m_state;
    }

    int ensemble_size()
    {
      return m_ensemble_size;
    }

    Vec ensemble_mean()
    {
      return m_ensemble_mean;
    }

    Mat ensemble_covariance()
    {
      return m_ensemble_covariance;
    }
    
    Vec observe(const Vec& x){
      return m_H * x;
    }

    Vec forecast(Vec& x, ForecastArgs&... args){
      return m_forecast(x, args...);
    }


  private:

    Mat gen_covariance_transform(const Mat& target_covariance) const
    {
      return target_covariance.ldlt().matrixL();
    }

    Mat gen_obs_perturbations()
    {
      auto rcovtransform = gen_covariance_transform(m_obs_covariance);

      std::random_device rd{};
      std::mt19937 gen{rd()};
      std::normal_distribution<> dis{0.0, 1.0};

      Mat A(m_H.rows(), m_ensemble_size);
      Vec epsilon(m_H.rows());

      for(auto i = 0; i < m_ensemble_size; ++i){
	for(auto e = 0; e < m_H.rows(); ++e){
	  epsilon[e] = dis(gen);
	}
	A.col(i) = epsilon;
      }

      return rcovtransform * A;
    }
      

    Mat gen_ensemble_perturbations(bool gen_cov_transform=true)
    {
      std::random_device rd{};
      std::mt19937 gen{rd()};
      std::normal_distribution<> dis{0.0, 1.0};

      Mat A(m_dimension,m_ensemble_size);
      Vec epsilon(m_dimension);

      if(gen_cov_transform){
	m_ensemble_transform = gen_covariance_transform(m_ensemble_covariance);
      }


      for(auto i = 0; i < m_ensemble_size; ++i){
	for(auto e = 0; e < m_dimension; ++e){
	  epsilon[e] = dis(gen);
	}
	A.col(i) = epsilon + m_ensemble_mean;
      }
      
      return  m_ensemble_transform * A;// * A;
    }

  /*std::pair<Vec,Mat> sample_mean_covariance(const std::vector<Vec>& sample)
    {
      Vec mu(sample[0].size());

      for(auto i = 0; i < m_ensemble_size; ++i){
	for(auto i = 0; i < sample[0].size(); ++i){
	  mu[i] += s[i];
	}
      }

      mu /= sample.size();
     
      Mat cov = Mat::Zero(sample.size(), sample.size());

      for(const auto& s : sample){
	auto delta = s - mu;
	cov.noalias() += delta * delta.transpose();
      }

      cov /= (sample.size() - 1);

      return std::make_pair(mu, cov);
      }*/
      
    //A is the result of a call to form_A
    Mat kalman_gain_A(const Mat& A)
    {
      auto V = m_H * A;

      auto vvr = V * V.transpose();// + m_obs_covariance;
      return A * V.transpose() * vvr.inverse();
    }   

    //copy ensemble here 
    Mat form_A(Mat ensemble)
    {
      for(auto i = 0; i < m_ensemble_size; ++i){
	ensemble.col(i) -= m_ensemble_mean;
      }

      return ensemble / std::sqrt(m_ensemble_size - 1);
    }

    Mat kalman_gain_ensemble(const Mat& ensemble)
    {
      auto A = form_A(ensemble);
      return kalman_gain_A(A);
    }

    //B matrix given an ensemble
    Mat ensemble_prior_covariance(const Mat& A)
    {
      return A * A.transpose();
    }
      

    Vec ensemble_member_update(const Mat& K, const Vec& perturbed_y, const Vec& xi)
    {
      auto err = perturbed_y - m_H * xi;

      return xi + K * err;
    }

    Mat ensemble_update(const Mat& K, const Mat& perturbed_yvals, const Mat& ensemble_xvals)
    {
      Mat updated_xvals(ensemble_xvals.rows(), ensemble_xvals.cols());
      for(auto i = 0; i < m_ensemble_size; ++i){
	updated_xvals.col(i) = ensemble_member_update(K, perturbed_yvals.col(i), ensemble_xvals.col(i));
      }
      return updated_xvals;
    }

    Mat ETKF_ensemble_update_mat(Mat& A)
    {
      auto V = m_H * A;
      auto Id = Mat::Identity(m_ensemble_size);
      auto emat = Id + V.transpose() * m_obs_covariance.inverse() * V;

      Eigen::SelfAdjointEigenSolver<Mat> esolver(emat);

      assert(esolver.info() == Eigen::Success);//, "ETKF eigendecomposition failed.");

      Mat Gamma = esolver.eigenvalues().asDiagonal();/* I + Gamma in the notes*/
      auto Q = esolver.eigenvectors();

      for(auto i = 0; i < Gamma.cols(); ++i){
	Gamma(i,i) = std::sqrt(Gamma(i,i));
      }

      return Q * Gamma * Q.transpose();
    }

    Mat ETKF_posterior_ensemble(Mat& A)
    {
      // A is prior ensemble
      auto X = ETKF_ensemble_update_mat(A);
      auto one = Vec::Ones(m_ensemble_size);
      auto aplus = A * one / m_ensemble_size;
      return A - aplus * one.transpose();
    }
      

    Mat ensemble_posterior_covariance(const Mat& B, const Mat& K)
    {
      return B - K * m_H * B;
    }

    

  public:

    void ETKF_filter(Vec& obs, ForecastArgs&... args)
    {
      for(auto i = 0; i < m_ensemble_size; ++i){
	  m_ensemble.col(i) = m_state;
	}

      m_ensemble += gen_ensemble_perturbations();

      m_ensemble = ETKF_posterior_ensemble(m_ensemble);

      m_state = m_ensemble.rowwise().mean();

      m_ensemble_mean = m_state;

      m_ensemble_covariance = m_ensemble * m_ensemble.transpose();
    }
      

    void filter(Vec& obs, ForecastArgs&... args)
    {
      // generate ensemble
      if(!m_initialized_ensemble){
	for(auto i = 0; i < m_ensemble_size; ++i){
	  m_ensemble.col(i) = m_state;
	}
	//m_initialized_ensemble = true;
      }
      m_ensemble += gen_ensemble_perturbations();
      /* forecast ensemble */
      for(auto i = 0; i < m_ensemble_size; ++i){
	Vec xstate = m_ensemble.col(i);
	m_ensemble.col(i) = m_forecast(xstate, args...);
      }

      m_ensemble_mean = m_ensemble.rowwise().mean();
      
      auto A = form_A(m_ensemble);
      m_background_covariance = A * A.transpose();
 
      Mat obs_perturbed = gen_obs_perturbations();

      for(auto i = 0; i < m_ensemble_size; ++i){
	obs_perturbed.col(i) += obs;
      }
      auto K = kalman_gain_A(A);
      A = ensemble_update(K, obs_perturbed / std::sqrt(m_ensemble_size-1), A);

      m_ensemble_covariance = A * A.transpose();

      /* get ensemble back */
      A *= std::sqrt(m_ensemble_size - 1);
      //add prior mean back and re-scale
      m_ensemble_mean = std::sqrt(m_ensemble_size - 1) * (A.rowwise().mean() + m_ensemble_mean);
      m_state = m_ensemble_mean;
    }
      
      

    
  };
  

}//namespace fastmath
#endif
