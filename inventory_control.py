import numpy as np
from sklearn.linear_model import Ridge
from itertools import chain
import pandas as pd
from scipy.stats import norm
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import time


def data_generation_inventory_control(n_episodes, shape_param,
                                      rate_param):  # using behavior policy= 0.5*control + 0.5*treatment
    trajectories = []
    horizon_len = np.random.gamma(shape=shape_param, scale=1 / rate_param, size=n_episodes)
    for k in range(n_episodes):
        episode = []
        steps = 0
        state = np.random.uniform(0, 1)
        while steps < horizon_len[k]:
            action = np.random.uniform(0, 1) - state
            demand = np.random.uniform(0, 1)
            next_state = max(state + action - demand, 0)
            reward = -next_state - 5 * max(demand - state - action, 0) - 2 * action
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state})
            state = next_state
            steps += 1

        trajectories.append(episode)

    return trajectories


def FQE_inventory_control(H, flatten_data, lambda_reg, type):
    if type == 'quadratic':
        d = 6
    elif type == 'cubic':
        d = 10
    elif type == 'fourth':
        d = 15
    theta_mat = np.zeros((d, H + 1))
    N = len(flatten_data)
    feature_mat_t = np.zeros((N, d))
    for h in range(H - 1, -1, -1):
        Q_tilde = []
        for n in range(N):
            reward = flatten_data[n]["reward"]
            next_state = flatten_data[n]["next_state"]
            state = flatten_data[n]["state"]
            action = flatten_data[n]["action"]
            if h == H - 1:
                feat_row = np.zeros(d)
                # print(f"state:{state}, action:{action}")
                if type == 'quadratic':
                    feature_row = [state ** 2, state * action, action ** 2, state, action,
                                   1]  # [(state + action)**2, state, action, 1]
                elif type == 'cubic':
                    feature_row = [state ** 3, (state ** 2) * action, state * (action ** 2), action ** 3, state ** 2,
                                   state * action, action ** 2, state, action, 1]
                elif type == 'fourth':
                    feature_row = [state ** 4,(state ** 3) * action, (state ** 2) * (action**2), state * (action **3), action**4, state ** 3, (state ** 2) * action, state * (action ** 2), action ** 3, state ** 2, state * action, action ** 2, state, action, 1]
                feature_mat_t[n, :] = feature_row
            elif h == H - 2:
                optimal_policy_action = -next_state + 0.5
                if type == 'quadratic':
                    feat_row = [next_state ** 2, next_state * optimal_policy_action, optimal_policy_action ** 2,
                                next_state, optimal_policy_action, 1]  # [0.25, next_state, -next_state + 0.5, 1]
                elif type == 'cubic':
                    feat_row = [next_state ** 3, (next_state ** 2) * optimal_policy_action,
                                next_state * (optimal_policy_action ** 2), optimal_policy_action ** 3, next_state ** 2,
                                next_state * optimal_policy_action, optimal_policy_action ** 2, next_state,
                                optimal_policy_action, 1]
                elif type == 'fourth':
                    feat_row = [next_state ** 4,(next_state ** 3) * optimal_policy_action, (next_state ** 2) * (optimal_policy_action**2), next_state * (optimal_policy_action **3), optimal_policy_action**4, next_state ** 3, (next_state ** 2) * optimal_policy_action,
                                next_state * (optimal_policy_action ** 2), optimal_policy_action ** 3, next_state ** 2,
                                next_state * optimal_policy_action, optimal_policy_action ** 2, next_state,
                                optimal_policy_action, 1]

            else:
                optimal_policy_action = -next_state + 0.75
                if type == 'quadratic':
                    feat_row = [next_state ** 2, next_state * optimal_policy_action, optimal_policy_action ** 2,
                                next_state, optimal_policy_action, 1]  # [9/16, next_state, -next_state + 0.75, 1]
                elif type == 'cubic':
                    feat_row = [next_state ** 3, (next_state ** 2) * optimal_policy_action,
                                next_state * (optimal_policy_action ** 2), optimal_policy_action ** 3, next_state ** 2,
                                next_state * optimal_policy_action, optimal_policy_action ** 2, next_state,
                                optimal_policy_action, 1]
                elif type == 'fourth':
                    feat_row = [next_state ** 4,(next_state ** 3) * optimal_policy_action, (next_state ** 2) * (optimal_policy_action**2), next_state * (optimal_policy_action **3), optimal_policy_action**4, next_state ** 3, (next_state ** 2) * optimal_policy_action,
                                next_state * (optimal_policy_action ** 2), optimal_policy_action ** 3, next_state ** 2,
                                next_state * optimal_policy_action, optimal_policy_action ** 2, next_state,
                                optimal_policy_action, 1]
            Q_tilde_ele = reward + np.dot(feat_row, theta_mat[:, h + 1])
            Q_tilde.append(Q_tilde_ele)

        model = Ridge(alpha=lambda_reg, fit_intercept=False)
        model.fit(feature_mat_t, Q_tilde)
        theta = model.coef_
        theta_mat[:, h] = theta
        # Q_mat[:,h] = theta #I * theta

    return theta_mat, feature_mat_t  # N*d


def value_return_estimation_optimal_policy_uniform_init(theta_0, type, H):  # q_table_0 = Q[0]
    if type == "quadratic":
        if H == 1:
            nu_0_pi_T = [1 / 3, -1 / 12, 1 / 12, 1 / 2, 0, 1]
        elif H > 1:
            nu_0_pi_T = [1 / 3, 1 / 24, 7 / 48, 1 / 2, 1 / 4, 1]
    elif type == "cubic":
        if H == 1:
            nu_0_pi_T = [1 / 4, -1 / 12, 1 / 24, 0, 1 / 3, -1 / 12, 1 / 12, 1 / 2, 0, 1]
        elif H > 1:
            nu_0_pi_T = [1 / 4, 0, 1 / 32, 5 / 64, 1 / 3, 1 / 24, 7 / 48, 1 / 2, 1 / 4, 1]
    elif type == "fourth":
        if H == 1:
            nu_0_pi_T = [1/5,-3/40, 1/30, -1/80, 1/80, 1/4, -1/12, 1/24, 0, 1/3, -1/12, 1/12, 1/2, 0, 1]
        elif H > 1:
            nu_0_pi_T = [1/5, -1/20, 7/240, -3/640, 3/640, 1/4, 1/24, 5/192, 0, 1/3, 1/24, 1/12, 1/2, 1/4, 1]

    value_estimate = np.dot(nu_0_pi_T, theta_0)

    return value_estimate


def feature_base_func(state, action, type):
    if type == "quadratic":
        rt = [state ** 2, state * action, action ** 2, state, action, 1]
    elif type == "cubic":
        rt = [state ** 3, (state ** 2) * action, state * (action ** 2), action ** 3, state ** 2, state * action,
              action ** 2, state, action, 1]
    elif type == "fourth":
        rt = [state ** 4,(state ** 3) * action, (state ** 2) * (action**2), state * (action **3), action**4, state ** 3, (state ** 2) * action, state * (action ** 2), action ** 3, state ** 2, state * action, action ** 2, state, action, 1]
    return rt


def compute_epsilon(Q_N, rewards, next_states, theta_est, H, type):
    num_transitions = len(rewards)
    epsilons = np.zeros((H, num_transitions))

    for h in range(H):
        for n in range(num_transitions):
            q_value_sn_an = Q_N[
                n, h]  # q_table[h,states[n] * n_actions + actions[n]]#q_table[states[n] * n_actions + actions[n], h]

            reward_n = rewards[n]
            next_state = next_states[n]

            if h == H - 2:
                optimal_policy_action = -next_state + 0.5
                v_h_plus_1 = feature_base_func(next_state, optimal_policy_action, type) @ theta_est[:, h + 1]
            elif h < H - 2:
                optimal_policy_action = -next_state + 0.75
                v_h_plus_1 = feature_base_func(next_state, optimal_policy_action, type) @ theta_est[:, h + 1]
            elif h == H - 1:
                v_h_plus_1 = 0

            # epsilon for current h and n
            epsilons[h, n] = q_value_sn_an - reward_n - v_h_plus_1

    return epsilons


def compute_Omega(errors, len_batch_eval, H, n_components, feature_table_t_K_2):
    # Initialize a dictionary to store Omega_est for each pair (h1, h2)
    Omega_est_list = {(h1, h2): np.zeros((n_components, n_components)) for h1 in range(H) for h2 in range(H)}

    # Gather the feature vectors for all transitions at once
    for h1 in range(H):
        phi_sa_h1 = feature_table_t_K_2 * (errors[h1:(h1 + 1), :].T)
        Omega_est_list[(h1, h1)] = np.dot(phi_sa_h1.T, phi_sa_h1)
        for h2 in range(h1 + 1, H):
            phi_sa_h2 = feature_table_t_K_2 * (errors[h2:(h2 + 1), :].T)
            Omega_est_list[(h1, h2)] = np.dot(phi_sa_h1.T, phi_sa_h2)
            # Omega_est_list[(h1, h2)] = np.einsum('ij,ik->jk', phi_sa_h1, phi_sa_h2)

    return {key: Omega_est_list[key] / len_batch_eval for key in Omega_est_list}


def compute_sigma_0_est(feature_table_t_K_2, len_batch_eval, H_bar):
    phi_sa_all = feature_table_t_K_2
    # np.vstack([feature_base_func(transition['state'], transition['action'], type) for transition in flattened_transitions])
    # Use np.einsum to compute the sum of outer products#Sigma_0_est = np.einsum('ij,ik->jk', phi_sa_all, phi_sa_all)
    Sigma_0_est = np.dot(phi_sa_all.T, phi_sa_all)
    # Normalize by K_A2 * H_bar
    Sigma_0_est /= (len_batch_eval * H_bar)

    return Sigma_0_est


def M_est_H_list(next_states, feature_matrix, n_transition, lambda_reg,H, type):
    n_components = np.shape(feature_matrix)[1]
    Sigma_hat = feature_matrix.T @ feature_matrix + lambda_reg * np.eye(n_components)
    Sigma_hat_inv = np.linalg.inv(Sigma_hat)
    # compute phi_pi and evaluate at collected trajectories, phi_pi_eval_at_data is N by n_components, each row is phi^pi(s)^T = sum_{a\in \chi} \phi(s,a)^T \pi(a|s)
    # phi_pi_eval_at_data_nextstate = np.zeros((n_transition, n_components))

    M_est_h_list = []
    phi_pi_eval_at_data_nextstate_check = []

    for h in range(H):
        phi_pi_eval_at_data_nextstate = np.zeros((n_transition, n_components))
        if h == H - 1:
            phi_pi_eval_at_data_nextstate = np.vstack(
                [feature_base_func(next_state, 0.5 - next_state, type) for next_state in next_states])
        elif h < H - 1:
            phi_pi_eval_at_data_nextstate = np.vstack(
                [feature_base_func(next_state, 0.75 - next_state, type) for next_state in next_states])

        phi_pi_eval_at_data_nextstate_check.append(phi_pi_eval_at_data_nextstate)

        phi_pi_phi_mat = np.dot(phi_pi_eval_at_data_nextstate.T, feature_matrix)
        M_est_h = Sigma_hat_inv @ phi_pi_phi_mat.T  # this is n_components * n_components
        M_est_h_list.append(M_est_h)

    return M_est_h_list  # , phi_pi_eval_at_data_nextstate_check, Phi_a


def compute_nu_h_est_T(H, M_H_matrix, type):
    # the initial_state_distribution is uniform (0,1)

    if type == "quadratic":
        if H == 1:
            nu_0_pi_T = np.array([1 / 3, -1 / 12, 1 / 12, 1 / 2, 0, 1])
        elif H > 1:
            nu_0_pi_T = np.array([1 / 3, 1 / 24, 7 / 48, 1 / 2, 1 / 4, 1])
    elif type == "cubic":
        if H == 1:
            nu_0_pi_T = np.array([1 / 4, -1 / 12, 1 / 24, 0, 1 / 3, -1 / 12, 1 / 12, 1 / 2, 0, 1])
        elif H > 1:
            nu_0_pi_T = np.array([1 / 4, 0, 1 / 32, 5 / 64, 1 / 3, 1 / 24, 7 / 48, 1 / 2, 1 / 4, 1])
    elif type == "fourth":
        if H == 1:
            nu_0_pi_T = np.array([1/5,-3/40, 1/30, -1/80, 1/80, 1/4, -1/12, 1/24, 0, 1/3, -1/12, 1/12, 1/2, 0, 1])
        elif H > 1:
            nu_0_pi_T = np.array([1/5, -1/20, 7/240, -3/640, 3/640, 1/4, 1/24, 5/192, 0, 1/3, 1/24, 1/12, 1/2, 1/4, 1])

    nu_h_T_est_list = [nu_0_pi_T]  # Start with ν_1^π

    # Iteratively compute ν_{h,est}^π for each h
    nu_h_T = nu_0_pi_T  # Initialize with ν_1^π
    for h in range(1, H):
        nu_h_T = np.dot(nu_h_T, M_H_matrix[h])
        nu_h_T_est_list.append(nu_h_T)

    return nu_h_T_est_list


def compute_sigma_squared(batch_estimation, batch_evaluation, H, H_bar, lambda_reg, type):
    """
    Compute sigma^2 using one batch for function estimation and another batch for evaluation.
    batch_estimation : A1. The batch of episodes used for function estimation (e.g., fitted Q-iteration).
    batch_evaluation : A2. The batch of episodes used for evaluating the estimated function and calculating sigma^2.
    Returns: sigma_squared : The estimated value of sigma^2 using the provided batches.
    """
    # Step 1: Estimate the function using batch_estimation (e.g., fitted Q-iteration)
    flattened_transitions_batch_estimation = list(chain.from_iterable(batch_estimation))
    theta_est, _ = FQE_inventory_control(H, flattened_transitions_batch_estimation, lambda_reg, type)
    theta_0 = theta_est[:, 0]

    # unpack batch_evaluation
    flattened_transitions_batch_eval = list(chain.from_iterable(batch_evaluation))
    # states_batch_eval = np.array([t["state"][0] if isinstance(t["state"], tuple) else t["state"] for t in flattened_transitions_batch_eval])
    # actions_batch_eval = np.array([t["action"] for t in flattened_transitions_batch_eval])
    next_states_batch_eval = np.array([t["next_state"] for t in flattened_transitions_batch_eval])
    rewards_batch_eval = np.array([t["reward"] for t in flattened_transitions_batch_eval])
    n_transition = len(flattened_transitions_batch_eval)

    feature_matrix_batch_eval = np.vstack(
        [feature_base_func(transition['state'], transition['action'], type) for transition in
         flattened_transitions_batch_eval])
    Q_N = feature_matrix_batch_eval @ theta_est
    # Step 2: Compute nu_h and Sigma_0 using batch_evaluation
    M_H_matrix = M_est_H_list(next_states_batch_eval, feature_matrix_batch_eval, n_transition, lambda_reg,H, type)
    # Compute nu_h for batch_evaluation
    nu_h_T = compute_nu_h_est_T(H, M_H_matrix, type)

    # Step 3: Compute Sigma0_est
    len_batch_eval = len(batch_evaluation)
    Sigma0_est = compute_sigma_0_est(feature_matrix_batch_eval, len_batch_eval,
                                     H_bar)  # Compute Sigma_0 for batch_evaluation
    n_components = np.shape(Sigma0_est)[1]
    Sigma0_est = Sigma0_est + lambda_reg * np.eye(n_components)

    # Step 4: Compute Omega using batch_evaluation
    epsilon = compute_epsilon(Q_N, rewards_batch_eval, next_states_batch_eval, theta_est, H, type)
    n_components = len(theta_0)
    Omega_est_list = compute_Omega(epsilon, len_batch_eval, H, n_components,
                                   feature_matrix_batch_eval)  # estimate_Omega(batch_evaluation, n_states, n_actions, epsilon,H)  # Compute Omega using batch_evaluation

    # Step 4: Calculate sigma_squared using the estimates from batch_estimation and batch_evaluation
    # Calculate term1: sum over h = 1 to H
    term1 = sum(
        (nu_h_T[h] @ np.linalg.inv(Sigma0_est) @ Omega_est_list[(h, h)] @ np.linalg.inv(Sigma0_est) @ nu_h_T[h].T) for h
        in range(H))

    # Calculate term2: sum over h1 < h2
    term2 = sum(2 * (
                nu_h_T[h1] @ np.linalg.inv(Sigma0_est) @ Omega_est_list[(h1, h2)] @ np.linalg.inv(Sigma0_est) @ nu_h_T[
            h2].T) for h1 in range(H) for h2 in range(h1 + 1, H))

    # Final sigma^2 estimate
    sigma_squared = (term1 + term2) / H_bar

    return sigma_squared, epsilon, Omega_est_list


def subsampled_bootstrapping_fqe(trajectories, v_hat_full, confidence_level, subset_size, B, lambda_reg,H, type):
    epsilon_list = []
    full_size = len(trajectories)

    # Step 3: Perform B bootstrap iterations
    for b in range(B):
        subset_indices = np.random.choice(len(trajectories), size=subset_size, replace=False)
        D_b = [trajectories[i] for i in subset_indices]

        flattened_transitions_D_b = list(chain.from_iterable(D_b))
        theta_est_D_b, _ = FQE_inventory_control(H, flattened_transitions_D_b, lambda_reg, type)
        theta_0_D_b = theta_est_D_b[:, 0]
        v_hat_subset = value_return_estimation_optimal_policy_uniform_init(theta_0_D_b, type, H)

        # Generate a resample set D^(b)*_K,s by resampling from D_b
        resample_indices = np.random.choice(len(D_b), size=full_size, replace=True)
        D_b_star = [D_b[i] for i in resample_indices]

        # Compute FQE estimator on the subset
        # q_table_subset, _ = fitted_q_evaluation_valid_state(D_b, H, Q_H, target_policy,target_epsilon,lambda_reg = 0.001, goal_reward=-1, n_actions=4, n_valid_states=38)

        # Compute FQE estimator on the resampled subset
        # v_hat_resample = fqe_estimator(D_b_star)
        flattened_transitions_D_b_star = list(chain.from_iterable(D_b_star))
        theta_est_star, _ = FQE_inventory_control(H, flattened_transitions_D_b_star, lambda_reg, type)
        theta_0_star = theta_est_star[:, 0]
        v_hat_resample = value_return_estimation_optimal_policy_uniform_init(theta_0_star, type, H)

        # Compute epsilon^(b)
        epsilon_b = v_hat_resample - v_hat_subset
        epsilon_list.append(epsilon_b)

    # Step 4: Estimate the variance of the FQE estimator
    epsilon_mean = np.mean(epsilon_list)
    variance = np.sum((np.array(epsilon_list) - epsilon_mean) ** 2) / (len(epsilon_list) - 1)

    # Step 5: Compute the confidence interval
    delta = 1 - confidence_level
    quantile_1 = np.percentile(epsilon_list, 100 * (delta / 2))
    quantile_2 = np.percentile(epsilon_list, 100 * (1 - delta / 2))
    confidence_interval = [v_hat_full - quantile_2, v_hat_full - quantile_1]

    return variance, confidence_interval, epsilon_list

