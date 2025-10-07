//
// Created by Khurram Javed on 2024-02-18.
//

#include "SwiftSarsa.h"
#include <vector>
#include <cmath>

SwiftSarsa::SwiftSarsa(int num_of_features, int num_of_actions, float lambda_init, float alpha_init,
                       float meta_step_size_init,
                       float eta_init,
                       float decay_init, float epsilon_init, float eta_min_init)
{
    this->alpha = alpha_init;
    this->w = std::vector<float>(num_of_features * num_of_actions, 0.0f);
    this->z = std::vector<float>(num_of_features * num_of_actions, 0.0f);
    this->z_delta = std::vector<float>(num_of_features * num_of_actions, 0.0f);
    this->delta_w = std::vector<float>(num_of_features * num_of_actions, 0.0f);

    for (int i = 0; i < num_of_actions; i++)
    {
        std::vector<int> temp;
        temp.reserve(num_of_features);
        for (int j = 0; j < num_of_features; j++)
        {
            temp.push_back(i * num_of_features + j);
        }
        this->action_feature_indices.push_back(temp);
    }


    this->h = std::vector<float>(num_of_features * num_of_actions, 0.0f);
    this->h_old = std::vector<float>(num_of_features * num_of_actions, 0.0f);
    this->h_temp = std::vector<float>(num_of_features * num_of_actions, 0.0f);
    this->beta = std::vector<float>(num_of_features * num_of_actions, logf(alpha_init));
    this->z_bar = std::vector<float>(num_of_features * num_of_actions, 0.0f);
    this->p = std::vector<float>(num_of_features * num_of_actions, 0.0f);
    this->last_alpha = std::vector<float>(num_of_features * num_of_actions, 0.0f);

    this->v_old = 0;
    this->lambda = lambda_init;
    this->v_delta = 0;
    this->eta = eta_init;
    this->decay = decay_init;
    this->epsilon = epsilon_init;
    this->meta_step_size = meta_step_size_init;
    this->eta_min = eta_min_init;
}


std::vector<float> SwiftSarsa::get_action_values(std::vector<std::pair<int, float>>& feature_indices) const
{
    auto action_values = std::vector<float>(this->action_feature_indices.size(), 0);
    for (int action_index = 0; action_index < this->action_feature_indices.size(); action_index++)
    {
        for (auto& index : feature_indices)
        {
            int real_index = this->action_feature_indices[action_index][index.first];
            action_values[action_index] += this->w[real_index] * index.second;
        }
    }
    return action_values;
}


void SwiftSarsa::do_computation_on_eligible_items(float value_of_the_chosen_action,
                                                  float gamma, float r)
{
    float delta = r + gamma * value_of_the_chosen_action - this->v_old;
    this->v_old = value_of_the_chosen_action;
    int position = 0;
    while
    (position < this->set_of_eligible_components.size())
    {
        int index = this->set_of_eligible_components[position];

        this->delta_w[index] = delta * this->z[index] - z_delta[index] * this->v_delta;
        this->w[index] += this->delta_w[index];
        this->beta[index] +=
            this->meta_step_size / (expf(this->beta[index])) * (delta - v_delta) * this->p[index];
        if (exp(this->beta[index]) < this->eta_min)
        {
            this->beta[index] = logf(this->eta_min);
        }
        if (exp(this->beta[index]) > this->eta || std::isinf(exp(this->beta[index])))
        {
            this->beta[index] = logf(this->eta);
        }
        this->h_old[index] = this->h[index];
        this->h[index] = this->h_temp[index] + delta * this->z_bar[index] - this->z_delta[index] * this->v_delta;
        this->h_temp[index] = this->h[index];
        z_delta[index] = 0;
        this->z[index] = gamma * this->lambda * this->z[index];
        this->p[index] = gamma * this->lambda * this->p[index];
        this->z_bar[index] = gamma * this->lambda * this->z_bar[index];
        if (this->z[index] <= this->last_alpha[index] * epsilon || this->z[index] == 0 || std::isinf(this->z[index]) ||
            std::isnan(this->z[index]))
        {
            this->z[index] = 0;
            this->p[index] = 0;
            this->z_bar[index] = 0;
            this->delta_w[index] = 0;
            this->beta[index] += logf(this->decay);
            this->set_of_eligible_components[position] = this->set_of_eligible_components[this->
                set_of_eligible_components.size() - 1];
            this->set_of_eligible_components.pop_back();
        }
        else
        {
            position++;
        }
    }
}


void SwiftSarsa::do_computation_on_active_features(std::vector<std::pair<int, float>>& feauture_indices_value_pairs)
{
    this->v_delta = 0;
    float tau = 0;

    for (auto& index : feauture_indices_value_pairs)
    {
        tau += expf(this->beta[index.first]) * index.second * index.second;
    }
    float E = this->eta;

    if (tau > this->eta)
    {
        E = tau;
    }

    float b = 0;
    for (auto& index : feauture_indices_value_pairs)
    {
        b += this->z[index.first] * index.second;
    }

    for (auto& index : feauture_indices_value_pairs)
    {
        if (z[index.first] == 0)
        {
            this->set_of_eligible_components.push_back(index.first);
        }
        this->v_delta += this->delta_w[index.first] * index.second;
        this->z_delta[index.first] = (this->eta / E) * expf(this->beta[index.first]) * index.second;
        this->last_alpha[index.first] = (this->eta / E) * expf(this->beta[index.first]);
        if ((this->eta / E) < 1)
        {
            this->h_temp[index.first] = 0;
            this->h[index.first] = 0;
            this->h_old[index.first] = 0;
            this->z_bar[index.first] = 0;
            this->beta[index.first] += logf(this->decay) * index.second * index.second;
        }

        this->z[index.first] += this->z_delta[index.first] * (1 - b);
        this->p[index.first] += this->h_old[index.first] * index.second;
        this->z_bar[index.first] += this->z_delta[index.first] * (1 - b - this->z_bar[index.first] * index.second);
        this->h_temp[index.first] = this->h[index.first] - this->h_old[index.first] * index.second * (this->z[index.
                first] - this->z_delta[index.first]) -
            this->h[index.first] * index.second * this->z_delta[index.first];
    }
}

float
SwiftSarsa::learn(std::vector<std::pair<int, float>>& feature_indices, float r, float gamma, int action)
{
    float value_of_the_chosen_action = 0;
    std::vector<std::pair<int, float>> features_indices_for_the_chosen_action;
    features_indices_for_the_chosen_action.reserve(feature_indices.size());
    for (auto& index : feature_indices)
    {
        int real_index = this->action_feature_indices[action][index.first];
        value_of_the_chosen_action += this->w[real_index] * index.second;
        features_indices_for_the_chosen_action.emplace_back(real_index, index.second);
    }

    this->do_computation_on_eligible_items(value_of_the_chosen_action, gamma, r);
    this->do_computation_on_active_features(features_indices_for_the_chosen_action);
    return value_of_the_chosen_action;
}

