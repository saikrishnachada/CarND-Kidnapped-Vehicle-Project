/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

//using std::string;
//using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
	num_particles = 100;
	default_random_engine gen;	
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
	  Particle current_par;
	  current_par.id = i;
	  current_par.x = dist_x(gen);
	  current_par.y = dist_y(gen);
	  current_par.theta = dist_theta(gen);
	  current_par.weight = 1.0;
	  
	  particles.push_back(current_par);
	  weights.push_back(current_par.weight);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
	default_random_engine gen;
	
	for (int i = 0; i < num_particles; i++) {
	  double prediction_x;
	  double prediction_y;
	  double prediction_theta;
	  //Instead of a hard check of 0, adding a check for very low value of yaw_rate
	  //  prediction_x = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
	  //  prediction_y = particles[i].y + (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
	  //  prediction_theta = particles[i].theta + (yaw_rate * delta_t);
	   if (fabs(yaw_rate) < 0.0001) {
	    prediction_x = particles[i].x + velocity * cos(particles[i].theta) * delta_t;
	    prediction_y = particles[i].y + velocity * sin(particles[i].theta) * delta_t;
	    prediction_theta = particles[i].theta;
	  } else {
	    prediction_x = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
	    prediction_y = particles[i].y + (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
	    prediction_theta = particles[i].theta + (yaw_rate * delta_t);
	  }
	  
	  normal_distribution<double> dist_x(prediction_x, std_pos[0]);
	  normal_distribution<double> dist_y(prediction_y, std_pos[1]);
	  normal_distribution<double> dist_theta(prediction_theta, std_pos[2]);
	  
	  particles[i].x = dist_x(gen);
	  particles[i].y = dist_y(gen);
	  particles[i].theta = dist_theta(gen);
	}
}


void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations, double sensor_range) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
	for (int i = 0; i < observations.size(); i++) {
	//Maximum distance can be square root of 2 times the range of sensor.
		double lowest_dist  = sensor_range * sqrt(2);
		int closest_landmark = -1;
		double observation_x = observations[i].x;
		double observation_y = observations[i].y;

		for (int j = 0; j < predicted.size(); j++) {
		  double prediction_x = predicted[j].x;
		  double prediction_y = predicted[j].y;
		  int prediction_id = predicted[j].id;
		  double current_dist = dist(observation_x, observation_y, prediction_x, prediction_y);

		  if (current_dist < lowest_dist ) {
		    lowest_dist  = current_dist;
		    closest_landmark = prediction_id;
		  }
		}
		observations[i].id = closest_landmark;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
// sensor_range 
  double normalize_weights = 0.0;
  for (int i = 0; i < num_particles; i++) {
    //Define a vector 'coordinate_transformation' which can store the observations transformed from cartesian coordinates to map co-ordinates
    vector<LandmarkObs> coordinate_transformation;
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs Tobs;
      Tobs.id = j;
      Tobs.x = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
      Tobs.y = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
      coordinate_transformation.push_back(Tobs);
    }

    //Filter map landmarks within sensor range
    vector<LandmarkObs> filtered_landmarks;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      Map::single_landmark_s current_landmark = map_landmarks.landmark_list[j];
      if ((fabs((particles[i].x - current_landmark.x_f)) <= sensor_range) && (fabs((particles[i].y - current_landmark.y_f)) <= sensor_range)) {
        filtered_landmarks.push_back(LandmarkObs {current_landmark.id_i, current_landmark.x_f, current_landmark.y_f});
      }
    }
    // Associate the transformed observations with the nearest landmark on the map
    dataAssociation(filtered_landmarks, coordinate_transformation, sensor_range);
    
	// updating the particles weight by applying multi-variate gaussian probability
    particles[i].weight = 1.0;

    double sigma_x_2 = pow(std_landmark[0], 2);
    double sigma_y_2 = pow(std_landmark[1], 2);
    double normalizer = (1.0/(2.0 * M_PI * std_landmark[0] * std_landmark[1]));
   
    for (int k = 0; k < coordinate_transformation.size(); k++) {
      double Tobs_x = coordinate_transformation[k].x;
      double Tobs_y = coordinate_transformation[k].y;
      double Tobs_id = coordinate_transformation[k].id;
      double prob = 1.0;

      for (int l = 0; l < filtered_landmarks.size(); l++) {
        double landmark_x = filtered_landmarks[l].x;
        double landmark_y = filtered_landmarks[l].y;
        double landmark_id = filtered_landmarks[l].id;

        if (Tobs_id == landmark_id) {
          prob = normalizer * exp(-1.0 * ((pow((Tobs_x - landmark_x), 2)/(2.0 * sigma_x_2)) + (pow((Tobs_y - landmark_y), 2)/(2.0 * sigma_y_2))));
          particles[i].weight *= prob;
        }
      }
    }
    normalize_weights += particles[i].weight;
  }
  //Normalize weights of all particles
  for (int i = 0; i < particles.size(); i++) {
    particles[i].weight /= normalize_weights;
    weights[i] = particles[i].weight;
  }
}


void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
	vector<Particle> particles_resampled;

	default_random_engine gen;	
	//Generate particle in random
	uniform_int_distribution<int> particle_index(0, num_particles - 1);
	int current_index = particle_index(gen);
	double weight_x = 0.0;	
	double max_weight = 2.0 * *max_element(weights.begin(), weights.end());
	
	for (int i = 0; i < particles.size(); i++) {
		uniform_real_distribution<double> random_weight(0.0, max_weight);
		weight_x += random_weight(gen);

	  while (weight_x > weights[current_index]) {
	    weight_x -= weights[current_index];
	    current_index = (current_index + 1) % num_particles;
	  }
	  particles_resampled.push_back(particles[current_index]);
	}
	particles = particles_resampled;
}


void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}