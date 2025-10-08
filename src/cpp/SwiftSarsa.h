
//
// Created by Khurram Javed on 2024-02-18.
//

#ifndef SWIFTSARSA_H
#define SWIFTSARSA_H

#include <vector>
#include <unordered_map>

class SwiftSarsa
{
    std::vector<int> set_of_eligible_components; // set of eligible items
    std::vector<std::vector<int>> action_feature_indices;
    std::vector<float> w;
    std::vector<float> z;
    std::vector<float> z_delta;
    std::vector<float> delta_w;
    float alpha;

    std::vector<float> h;
    std::vector<float> h_old;
    std::vector<float> h_temp;
    std::vector<float> beta;
    std::vector<float> z_bar;
    std::vector<float> p;

    std::unordered_map<int, int> indices_map;
    float epsilon;

    std::vector<float> last_alpha;
    float v_delta;
    float v_old;
    float meta_step_size;
    float eta;
    float decay;
    float lambda;
    float eta_min;
    void do_computation_on_eligible_items(float value_of_action_taken, float gamma, float r);
    void do_computation_on_active_features(std::vector<std::pair<int, float>>& feauture_indices_value_pairs);

public:
    std::vector<float> get_action_values(std::vector<std::pair<int, float>>& indices) const;
    float learn(std::vector<std::pair<int, float>>& indices, float reward, float gamma,
                int action);
    SwiftSarsa(int num_of_features, int num_of_actions, float lambda, float alpha,
               float meta_step_size,
               float eta,
               float decay, float epsilon = 1e-4, float eta_min = 1e-8);
};


class SwiftSarsaBinaryFeatures
{
    std::vector<int> set_of_eligible_components; // set of eligible items
    std::vector<std::vector<int>> action_feature_indices;
    std::vector<float> w;
    std::vector<float> z;
    std::vector<float> z_delta;
    std::vector<float> delta_w;
    float alpha;

    std::vector<float> h;
    std::vector<float> h_old;
    std::vector<float> h_temp;
    std::vector<float> beta;
    std::vector<float> z_bar;
    std::vector<float> p;

    std::unordered_map<int, int> indices_map;
    float epsilon;

    std::vector<float> last_alpha;
    float v_delta;
    float v_old;
    float meta_step_size;
    float eta;
    float decay;
    float lambda;
    float eta_min;
    void do_computation_on_eligible_items(float value_of_action_taken, float gamma, float r);
    void do_computation_on_active_features(std::vector<int>& feauture_indices_value_pairs);

public:
    std::vector<float> get_action_values(std::vector<int>& indices) const;
    float learn(std::vector<int>& indices, float reward, float gamma,
                int action);
    SwiftSarsaBinaryFeatures(int num_of_features, int num_of_actions, float lambda, float alpha,
               float meta_step_size,
               float eta,
               float decay, float epsilon = 1e-4, float eta_min = 1e-8);
};


#endif // SWIFTSARSA_H
