
#include <random>
#include <algorithm>
#include <random>
#include <iostream>
#include <tuple>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using std::normal_distribution;
using std::default_random_engine;

using vector_t = std::vector<double>;

default_random_engine gen;


using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // DONE: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    num_particles = 200;
    weights.resize(num_particles, 1.0f);
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // Add random Gaussian noise to each particle.
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (unsigned i = 0; i < num_particles; i++) {
        Particle p;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.id = i;
        p.weight = 1.0f;
        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // DONE: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    cout << "Begin prediction" << endl;
    double x_pred_mean;
    double y_pred_mean;
    double theta_mean;

    for (int i = 0; i < num_particles; i++) {
        //calculate predicted x, y, theta
        if (fabs(yaw_rate) < 0.00001) {
            x_pred_mean = particles[i].x + velocity * delta_t * cos(particles[i].theta);
            y_pred_mean = particles[i].y + velocity * delta_t * sin(particles[i].theta);
        } else {
            x_pred_mean = particles[i].x + (velocity / yaw_rate) *
                                           (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            y_pred_mean = particles[i].y + (velocity / yaw_rate) *
                                           (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
            theta_mean = particles[i].theta + yaw_rate * delta_t;
        }

        //define generator with predicted x, y, theta as mean and with std_pos noise
        normal_distribution<double> dist_x(x_pred_mean, std_pos[0]);
        normal_distribution<double> dist_y(y_pred_mean, std_pos[1]);
        normal_distribution<double> dist_psi(theta_mean, std_pos[2]);

        //predict particle with noise
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_psi(gen);

    }
    cout << "End prediction" << endl;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // DONE: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    double min_distance;

    for (int i = 0; i < observations.size(); i++) {
        LandmarkObs observation = observations[i];
        // initialize min_distance to maximum value
        min_distance = INFINITY;

        // landmark_id of the closest current_observation
        int closest_landmark_id = -1;


        for (unsigned int j = 0; j < predicted.size(); j++) {
            LandmarkObs prediction = predicted[j];

            // get distance between current and predicted landmarks
            double cur_dist = dist(observation.x, observation.y, prediction.x, prediction.y);

            // find the nearest landmark based on cur_dist
            if (cur_dist < min_distance) {
                min_distance = cur_dist;
                closest_landmark_id = prediction.id;
            }
        }

        observations[i].id = closest_landmark_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // DONE: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    cout << "Begin updateWeights" << endl;
    // Bringing out all the landmark based calculation out of loop
    // as it is kind of remains same
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    double gauss_norm = (1 / (2 * M_PI * std_x * std_y));
    double std_x_power = 2 * pow(std_x, 2);
    double std_y_power = 2 * pow(std_y, 2);

    for (int i = 0; i < num_particles; i++) {

        // get the particle x, y coordinates
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;

        // vector to hold the map landmark locations predicted within sensor range of the particle
        vector<LandmarkObs> predictions;

        for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

            // get id and x,y coordinates
            float lm_x = map_landmarks.landmark_list[j].x_f;
            float lm_y = map_landmarks.landmark_list[j].y_f;
            int lm_id = map_landmarks.landmark_list[j].id_i;

            if (dist(p_x, p_y, lm_x, lm_y) < sensor_range) {
                predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
            }
        }

        // create and populate a copy of the list of observations transformed from vehicle coordinates to map coordinates
        vector<LandmarkObs> transformed_observations;
        for (unsigned int j = 0; j < observations.size(); j++) {
            double t_x = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + p_x;
            double t_y = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + p_y;
            transformed_observations.push_back(LandmarkObs{observations[j].id, t_x, t_y});
        }

        // perform dataAssociation for the predictions and transformed_observations on current particle
        dataAssociation(predictions, transformed_observations);

        // reset weight
        particles[i].weight = 1.0;

        for (unsigned int j = 0; j < transformed_observations.size(); j++) {

            // observation and prediction coordinates
            double obs_x, obs_y, pr_x, pr_y;
            obs_x = transformed_observations[j].x;
            obs_y = transformed_observations[j].y;

            int associated_prediction_id = transformed_observations[j].id;

            // get the x,y coordinates of the prediction for the current observation
            // TODO: optimize further by using hashmap
            for (unsigned int k = 0; k < predictions.size(); k++) {
                if (predictions[k].id == associated_prediction_id) {
                    pr_x = predictions[k].x;
                    pr_y = predictions[k].y;
                }
            }

            // calculate weight for this observation with multivariate Gaussian
            double exponent = exp(-(pow(pr_x - obs_x, 2) / std_x_power +
                                    (pow(pr_y - obs_y, 2) / std_y_power)));
            double obs_w = gauss_norm * exponent;
            // particles final weight
            particles[i].weight *= obs_w;
        }
    }
    cout << "End updateWeights" << endl;
}

void ParticleFilter::resample() {
    // DONE: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    cout << "Begin resample" << endl;
    vector<Particle> resampled_particles;

    // get all the weights from particles
    vector<double> weights;
    for (int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
    }

    discrete_distribution<> distribution(weights.begin(), weights.end());

    for (int i = 0; i < particles.size(); i++) {
        int weighted_index = distribution(gen);
        resampled_particles.push_back(particles[weighted_index]);
    }

    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
