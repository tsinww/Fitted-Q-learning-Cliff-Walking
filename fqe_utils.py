import numpy as np
from sklearn.linear_model import Ridge
from itertools import chain
import random

# Policy definition
# Define optimal-target policy #\pi_h
def optimal_policy(state, q_table_h,n_actions):
    q_values = [q_table_h[state * n_actions + a] for a in range(n_actions)]
    rt_action = np.zeros(n_actions)
    rt_action[np.argmax(q_values)]=1
    return rt_action

def epsilon_greedy_policy_data_generation(state, behavior_epsilon, Q_optimal,env):
    if random.random() < behavior_epsilon:
        action = env.action_space.sample()  # Sample a random action from the action space
        #custom_action = gym_to_custom_action_mapping[action]
    else:
        max_indices = np.where(Q_optimal[state] == np.max(Q_optimal[state]))[0]
        if len(max_indices) == 1:
            action = np.argmax(Q_optimal[state])
        else:
            action = np.random.choice(max_indices)
        # Use Q-table to select the best action (greedy action)
        #custom_action = np.argmax(Q_optimal[state])
    return action

def epsilon_greedy_policy_prob_distribution(state, Q_optimal, n_actions,policy_epsilon):
    pi_action_state = np.zeros(n_actions)
    if state <=36:
        max_indices = np.where(Q_optimal[state] == np.max(Q_optimal[state]))[0]
        for action in range(n_actions):
            if action in max_indices:
                pi_action_state[action] = (1-policy_epsilon)/len(max_indices) + policy_epsilon/n_actions
            else:
                pi_action_state[action] = policy_epsilon/n_actions
    return pi_action_state


# encode transition matrix and reward matrix # U = 0, R = 1, D = 2, L = 3.

# deterministic transition
def transition_matrix_valid_state(n_rows=4, n_cols=12, n_actions=4):
    n_states = n_rows * n_cols
    n_valid_states = n_states - 10
    # rewards = np.array([[[None for _ in range(n_valid_states)] for _ in range(n_actions)] for _ in range(n_valid_states)])#-1 * np.ones((n_valid_states, n_actions, n_valid_states))#here rewards is a vector function of next_states, state, action
    transitions = np.zeros((n_valid_states, n_actions), dtype=int)
    # penalty = -100

    cliff_row = 3

    start_state = 36
    goal_state = 37

    for state in range(n_valid_states):
        row, col = divmod(state, n_cols)
        transitions[state, 0] = max(n_cols * (row - 1) + col, col)
        transitions[state, 1] = n_cols * row + min(col + 1, n_cols - 1)
        transitions[state, 2] = min(n_cols * (row + 1) + col, n_cols * (n_rows - 1) + col)
        transitions[state, 3] = n_cols * row + max(col - 1, 0)
        if state == start_state:
            transitions[state, 1] = state
        elif row == cliff_row - 1 and 1 <= col <= 10:
            transitions[state, 2] = start_state
        elif row == cliff_row - 1 and col == 11:
            transitions[state, 2] = goal_state  # start_state
            # rewards[state,1, start_state] = -1
        # else:
        #    rewards[state,1, transitions[state,1]] = -1
        #    rewards[state,3, transitions[state,3]] = -1

    return transitions  # , rewards


def rewards_matrix_valid_state(epsilon_transition, goal_reward, n_valid_states,
                               n_actions=4):  # computing the reward function
    penalty = -100

    rewards = -1 * np.ones((n_valid_states,
                            n_actions))  # np.array([[[None for _ in range(n_valid_states)] for _ in range(n_actions)] for _ in range(n_valid_states)])
    start_state = 36
    # goal_reward = -1
    for state in range(n_valid_states):
        if state == start_state:
            rewards[state, [0, 2, 3]] = -1 * (1 - epsilon_transition / 4) + penalty * epsilon_transition / 4
            rewards[state, 1] = (1 - 3 * epsilon_transition / 4) * penalty - 3 * epsilon_transition / 4
            # rewards[state, 0:3] = -1*(1-epsilon_transition) - 3*epsilon_transition/4 + penalty * epsilon_transition/4
            # rewards[state, 3] = (1-epsilon_transition)*penalty - 3*epsilon_transition/4 + penalty * epsilon_transition/4
        elif 25 <= state <= 34:  # row == cliff_row-1 and 1 <= col <= 10:
            rewards[state, [0, 1, 3]] = -1 * (1 - epsilon_transition / 4) + penalty * epsilon_transition / 4
            rewards[state, 2] = (1 - 3 * epsilon_transition / 4) * penalty - 3 * epsilon_transition / 4
        elif state == 35:
            rewards[state, [0, 1, 3]] = -1 * (1 - epsilon_transition / 4) + goal_reward * epsilon_transition / 4
            rewards[state, 2] = - 3 * epsilon_transition / 4 + (1 - 3 * epsilon_transition / 4) * goal_reward

    return rewards


def GT_Q_values_valid_states(epsilon_transition, H, goal_reward, n_actions, n_valid_states, gamma=1):
    transitions = transition_matrix_valid_state()
    rewards = rewards_matrix_valid_state(epsilon_transition, goal_reward, n_valid_states, n_actions=4)
    # uniformly randon select with prob epsilon_transition
    # n_valid_states = n_states - 11
    Q = np.zeros((H + 1, n_valid_states, n_actions))
    optimal_action_multiple = np.empty((H, n_valid_states), dtype=object)
    # Q_old = np.copy(Q)

    for h in range(H - 1, -1, -1):
        # Q_old = Q[:,:, h+1]
        for state in range(n_valid_states - 1):
            next_states_potential = transitions[state]
            for action in range(n_actions):
                next_states_deter = next_states_potential[action]
                Q_ele = np.max(Q[h + 1, next_states_deter, :]) * (1 - epsilon_transition)
                for next_state in next_states_potential:
                    Q_ele += np.max(Q[h + 1, next_state, :]) * epsilon_transition / 4

                Q[h, state, action] = rewards[state, action] + gamma * Q_ele

            # print(np.where(Q[state,:,h] == np.max(Q[state,:,h]))[0])
            optimal_action_multiple[h, state] = np.where(Q[h, state, :] == np.max(Q[h, state, :]))[0]

        # Update Q_old for the next iteration
        # Q_old = np.copy(Q)

    return Q, optimal_action_multiple


def value_return_estimation(q_table_0, initial_state_distribution, target_policy, policy_epsilon):  # q_table_0 = Q[0]
    value_estimate = 0
    expected_q_set = []
    n_actions = 4
    q_vec_0 = q_table_0.flatten()  # vectorize(q_table_0)
    for state, prob in initial_state_distribution.items():
        index_set = state * n_actions + np.arange(n_actions)
        expected_q_values = np.dot(q_vec_0[index_set], target_policy(state, q_table_0, n_actions, policy_epsilon))
        # expected_q_values = q_table_0[index]#np.dot(q_values_at_all_actions_given_state, target_policy(state, q_table_0,n_actions))
        expected_q_set.append(expected_q_values)
        # print(f"GT expected_q_values: {expected_q_values}")
        # taking expectation again w.r.t. initial distribution
        value_estimate += prob * expected_q_values
        # print(value_estimate)

    return value_estimate, expected_q_set


def diagram_action(optimal_action_multiple, H, n_valid_states):
    n_rows = 4
    n_cols = 12

    action_labels_plot = {0: '\u2191', 1: '\u2192', 2: '\u2193',
                          3: '\u2190'}  # {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}

    # Reshape the best actions into a 4x12 grid
    best_actions_grid = np.full((H, n_rows, n_cols), '', dtype=object)

    for h in range(H):
        for state in range(n_valid_states - 1):
            row, col = divmod(state, n_cols)
            best_actions_for_state = optimal_action_multiple[h, state]
            best_actions_grid[h, row, col] = ', '.join(
                [action_labels_plot[action] for action in best_actions_for_state])

    return best_actions_grid


def diagram_action_from_q_values(Q_mat, n_valid_states):
    n_rows = 4
    n_cols = 12

    action_labels_plot = {0: '\u2191', 1: '\u2192', 2: '\u2193',
                          3: '\u2190'}  # {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}

    optimal_action_multiple = np.empty((n_valid_states), dtype=object)

    for state in range(n_valid_states):
        optimal_action_multiple[state] = np.where(Q_mat[state] == np.max(Q_mat[state]))[0]

    # Reshape the best actions into a 4x12 grid
    best_actions_grid = np.full((n_rows, n_cols), '', dtype=object)
    for state in range(n_valid_states - 1):
        row, col = divmod(state, n_cols)
        best_actions_for_state = optimal_action_multiple[state]
        best_actions_grid[row, col] = ', '.join([action_labels_plot[action] for action in best_actions_for_state])

    return best_actions_grid


def stochastic_transition_valid_state_absorbing(env, state, action, transition_epsilon):
    transitions = transition_matrix_valid_state()

    if state == 37:
        reward = 0
        next_state = 37
    # If the random number is less than epsilon, take a random action
    elif state < 37:
        if random.random() < transition_epsilon:
            action = env.action_space.sample()  # Sample a random action from Gym's action space
            next_state = transitions[state, action]
            # terminated = False
            # truncated = False
            reward = -1
            if state == 35 and action == 2:
                reward = -1
                # terminated = True
            elif 25 <= state <= 34 and action == 2:
                reward = -100
            elif state == 36 and action == 1:
                reward = -100
        else:
            # next_state, reward, terminated, truncated, info = env.step(action)
            next_state = transitions[state, action]
            reward = -1
            if 25 <= state <= 34 and action == 2:  # State 25 to 34, Action DOWN
                next_state = 36  # Move to the cliff state (states 37 to 46)
                reward = -100  # Set the reward to -100 for falling into the cliff
                # terminated = False  # The episode should not terminate in the cliff state
                # truncated = False  # We don't want truncation in the cliff state either
            elif state == 35 and action == 2:
                reward = -1
                next_state = 37  # Stay in the cliff state
                # terminated = True  # The episode should not terminate in the cliff state
                # truncated = False  # We don't want truncation in the cliff state either
            elif state == 36 and action == 1:
                next_state = 36  # Move to the cliff state (states 37 to 46)
                reward = -100  # Set the reward to -100 for falling into the cliff
                # terminated = False  # The episode should not terminate in the cliff state
                # truncated = False  # We don't want truncation in the cliff state either

    return next_state, reward  # , terminated, truncated


def collect_trajectories_valid_state_absorbing(env, n_episodes, shape_param, rate_param, transition_epsilon,
                                               behavior_policy, Q_optimal, behavior_epsilon):
    # Q_optimal, behavior_epsilon
    trajectories = []
    cliff_reached = 0  # Counter for how many times the agent reaches a cliff state
    H_data = np.random.gamma(shape=shape_param, scale=1 / rate_param, size=n_episodes)

    for k in range(n_episodes):
        state = env.reset()
        episode = []
        # done = False
        # truncated = False
        steps = 0

        # Check if state is a tuple and handle it accordingly
        if isinstance(state, tuple):  # e.g., (row, col) in a grid environment
            state = state[0]

        while steps < H_data[k]:
            # action = env.action_space.sample()  # Sample action from the policy (random or deterministic)
            # Epsilon-greedy policy: with probability epsilon, take a random action, else take the best action
            action = behavior_policy(state, behavior_epsilon, Q_optimal,env)
            next_state, reward = stochastic_transition_valid_state_absorbing(env, state, action, transition_epsilon)

            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state})
            state = next_state
            steps += 1

            # if terminated == True:
            #    break

        trajectories.append(episode)

    # Print the number of times cliff states were reached
    # print(f"Cliff states reached {cliff_reached} times out of {n_episodes} episodes.")

    return trajectories


def fitted_q_evaluation_valid_state(trajectories, H, Q_H, target_policy, target_epsilon, lambda_reg, goal_reward=-1,
                                    n_actions=4, n_valid_states=38):
    q_table = np.zeros((H + 1, n_valid_states * n_actions))
    theta_hat = [None for _ in range(H + 1)]
    theta_hat[H] = np.zeros((n_valid_states) * n_actions)

    flatten_trajectories = list(chain.from_iterable(trajectories))
    feature_matrix = []  # List to hold feature vectors for all state-action pairs in the data
    for h in range(H, 0, -1):
        targets = []  # List to hold regression targets (y)
        for transition in flatten_trajectories:
            state = transition["state"]
            action = transition["action"]
            reward = transition["reward"]
            next_state = transition["next_state"]

            Q_next = [theta_hat[h][next_state * n_actions + a] for a in range(n_actions)]
            target = reward + np.dot(Q_next, target_policy(next_state, Q_H[h], n_actions, target_epsilon))
            targets.append(target)

            if h == H:
                phi_sa = np.zeros(n_valid_states * n_actions)
                phi_sa[state * n_actions + action] = 1
                feature_matrix.append(phi_sa)

        feature_matrix = np.array(feature_matrix)
        targets = np.array(targets)
        model = Ridge(alpha=lambda_reg, fit_intercept=False)
        model.fit(feature_matrix, targets)

        theta_hat[h - 1] = model.coef_

        q_table[h - 1] = theta_hat[h - 1]  # np.dot(np.eye((n_valid_states) * n_actions), model.coef_)

    return q_table, feature_matrix


def data_collected_embedded(flattened_transitions, features_table_t=np.eye(152)):
    n_actions = 4
    # flattened_transitions = list(chain.from_iterable(trajectories))
    features_from_data = []
    for transition in flattened_transitions:
        state = transition["state"][0] if isinstance(transition["state"], tuple) else transition["state"]
        index = state * n_actions + transition["action"]
        features_from_data.append(features_table_t[index])
    return np.array(features_from_data)


# Compute the expectation of phi^pi(s') given (s, a) E(phi^\pi(s')|s,a), here the output is for all (s,a) \in \chi
def expectation_phi_pi(states, actions, next_states, features_table_t, target_policy, Q_optimal, policy_epsilon,
                       lambda_reg):
    # q_table_h is the h_th column of q_table, q_table is NsNa * H
    n_components = np.shape(features_table_t)[1]
    n_transition = len(states)
    n_actions = 4
    features_matrix = np.array([features_table_t[states[k] * n_actions + actions[k]] for k in
                                range(n_transition)])  # Shape: (n_transition, n_components)
    Sigma_hat = features_matrix.T @ features_matrix + lambda_reg * np.eye(n_components)
    Sigma_hat_inv = np.linalg.inv(Sigma_hat)

    # compute phi_pi and evaluate at collected trajectories, phi_pi_eval_at_data is N by n_components, each row is phi^pi(s)^T = sum_{a\in \chi} \phi(s,a)^T \pi(a|s)
    phi_pi_eval_at_data_nextstate = np.zeros((n_transition, n_components))
    Pi_action_all = np.array([target_policy(next_state, Q_optimal, n_actions, policy_epsilon) for next_state in
                              next_states])  # dimension: number of next_states from collected data * Na
    for action in range(n_actions):
        Phi_a = []
        for next_state in next_states:
            if next_state > 37:
                Phi_a.append(np.zeros(n_components))
            else:
                Phi_a.append(features_table_t[next_state * n_actions + action])
        Phi_a = np.array(Phi_a)
        # Phi_a = np.array([features_table_t[next_state * n_actions + action] for next_state in next_states])
        Pi_a = Pi_action_all[:, action:(action + 1)]  # Pi_action_all[:, action].reshape(-1, 1)
        phi_pi_eval_at_data_nextstate += Phi_a * Pi_a

    # features_matrix = np.array([features_table_t[states[k] * n_actions + actions[k]] for k in range(n_transition)])  # Shape: (n_transition, n_components)
    # Perform batch outer product and sum along the batch dimension
    # phi_pi_phi_mat = np.einsum('ij,ik->jk', phi_pi_eval_at_data_nextstate, features_matrix)
    phi_pi_phi_mat = np.dot(phi_pi_eval_at_data_nextstate.T, features_matrix)

    return phi_pi_phi_mat @ Sigma_hat_inv @ features_table_t.T  # this is n_components * NsNa


def M_est_H_list(next_states, feature_matrix, n_transition, Q_H, target_epsilon, lambda_reg,target_policy,H):
    n_actions = 4
    n_components = np.shape(feature_matrix)[1]
    Sigma_hat = feature_matrix.T @ feature_matrix + lambda_reg * np.eye(n_components)
    Sigma_hat_inv = np.linalg.inv(Sigma_hat)
    # compute phi_pi and evaluate at collected trajectories, phi_pi_eval_at_data is N by n_components, each row is phi^pi(s)^T = sum_{a\in \chi} \phi(s,a)^T \pi(a|s)
    # phi_pi_eval_at_data_nextstate = np.zeros((n_transition, n_components))

    M_est_h_list = []
    phi_pi_eval_at_data_nextstate_check = []

    Phi_all_a = []
    # for action in range(n_actions):
    #    Phi_a = []
    #    for next_state in next_states:
    #        index = next_state * n_actions + action
    #        phi_next_state = np.zeros(n_components)
    #        phi_next_state[index] = 1
    #        Phi_a.append(phi_next_state)
    #    Phi_a = np.array(Phi_a)
    #    Phi_all_a.append(Phi_a)

    for h in range(H):
        phi_pi_eval_at_data_nextstate = np.zeros((n_transition, n_components))
        Pi_action_all = np.array([target_policy(next_state, Q_H[h], n_actions, target_epsilon) for next_state in
                                  next_states])  # dimension: number of next_states from collected data * Na
        for action in range(n_actions):
            if h == 0:
                Phi_a = []
                for next_state in next_states:
                    index = next_state * n_actions + action
                    phi_next_state = np.zeros(n_components)
                    phi_next_state[index] = 1
                    Phi_a.append(phi_next_state)
                Phi_a = np.array(Phi_a)
                Phi_all_a.append(Phi_a)
            # Phi_a = np.array([features_table_t[next_state * n_actions + action] for next_state in next_states])
            Pi_a = Pi_action_all[:, action:(action + 1)]  # Pi_action_all[:, action].reshape(-1, 1)
            phi_pi_eval_at_data_nextstate += Phi_all_a[action] * Pi_a

        phi_pi_eval_at_data_nextstate_check.append(phi_pi_eval_at_data_nextstate)

        # features_matrix = np.array([features_table_t[states[k] * n_actions + actions[k]] for k in range(n_transition)])  # Shape: (n_transition, n_components)
        # Perform batch outer product and sum along the batch dimension
        # phi_pi_phi_mat = np.einsum('ij,ik->jk', phi_pi_eval_at_data_nextstate, features_matrix)
        phi_pi_phi_mat = np.dot(phi_pi_eval_at_data_nextstate.T, feature_matrix)
        # print(h)
        # Compute Phi_pi for each h
        # Phi_pi = expectation_phi_pi(states, actions, next_states, features_table_t, target_policy, Q_H[h],policy_epsilon,lambda_reg)
        # Compute M_est_h
        M_est_h = Sigma_hat_inv @ phi_pi_phi_mat.T  # this is n_components * n_components
        M_est_h_list.append(M_est_h)

    return M_est_h_list  # , phi_pi_eval_at_data_nextstate_check, Phi_a


def compute_nu_h_est_T(H, M_H_matrix, initial_state_distribution, target_policy, Q_0, target_epsilon, n_valid_states,
                       n_components):
    nu_0_pi_T = np.zeros(n_components)
    n_actions = 4
    # features_table_t = np.eye(n_valid_states*n_actions)

    # Iterate over each state and action
    for s in range(n_valid_states):
        xi_s = initial_state_distribution.get(s, 0)
        ###########
        pi_s = target_policy(s, Q_0, n_actions, target_epsilon)
        for a in range(n_actions):
            phi_sa = np.zeros(n_components)
            phi_sa[s * n_actions + a] = 1
            # phi_sa = features_table_t[s * n_actions + a, :]
            nu_0_pi_T += xi_s * phi_sa * pi_s[a]

    nu_h_T_est_list = [nu_0_pi_T]  # Start with ν_1^π

    # Iteratively compute ν_{h,est}^π for each h
    nu_h_T = nu_0_pi_T  # Initialize with ν_1^π
    for h in range(1, H):
        nu_h_T = np.dot(nu_h_T, M_H_matrix[h])
        nu_h_T_est_list.append(nu_h_T)

    return nu_h_T_est_list


def compute_epsilon(q_table, rewards, next_states, states, actions, H, n_actions, target_policy, Q_H, policy_epsilon):
    num_transitions = len(states)
    epsilons = np.zeros((H, num_transitions))

    for h in range(H):
        for n in range(num_transitions):
            # Retrieve Q_h^{pi}(s_n, a_n) from q_table
            q_value_sn_an = q_table[
                h, states[n] * n_actions + actions[n]]  # q_table[states[n] * n_actions + actions[n], h]

            # Retrieve reward r_n
            reward_n = rewards[n]

            # Calculate V_{h+1}^{pi}(s_n') as the weighted sum of Q_{h+1}^{pi}(s_n', a) by policy probabilities
            next_state = next_states[n]
            if next_state == 47:
                v_h_plus_1 = 0
            else:
                #######
                pi_a_given_sn_prime = target_policy(next_state, Q_H[h + 1], n_actions, policy_epsilon)
                q_values_next_state = q_table[h + 1, next_state * n_actions: (next_state + 1) * n_actions]  # q_table[next_state * n_actions: (next_state + 1) * n_actions, h+1]  # Q_{h+1}^{pi}(s_n', a) for all a
                # print(q_values_next_state)
                # print("next state")
                # print(next_state)
                v_h_plus_1 = np.dot(q_values_next_state, pi_a_given_sn_prime)

            # Calculate epsilon for the current h and n
            epsilons[h, n] = q_value_sn_an - reward_n - v_h_plus_1

    return epsilons


def compute_Omega(flattened_transitions, features_table_t, errors, len_batch_eval, H, n_components):
    n_actions = 4
    # Collect indices for each (s, a) pair
    indices = [
        (transition["state"][0] if isinstance(transition["state"], tuple) else transition["state"]) * n_actions +
        transition["action"]
        for transition in flattened_transitions
    ]
    # Initialize a dictionary to store Omega_est for each pair (h1, h2)
    Omega_est_list = {(h1, h2): np.zeros((n_components, n_components)) for h1 in range(H) for h2 in range(H)}

    # Gather the feature vectors for all transitions at once
    for h1 in range(H):
        phi_sa_h1 = features_table_t[indices, :] * (errors[h1:(h1 + 1), :].T)
        Omega_est_list[(h1, h1)] = np.dot(phi_sa_h1.T, phi_sa_h1)
        for h2 in range(h1 + 1, H):
            phi_sa_h2 = features_table_t[indices, :] * (errors[h2:(h2 + 1), :].T)
            Omega_est_list[(h1, h2)] = np.dot(phi_sa_h1.T, phi_sa_h2)
            # Omega_est_list[(h1, h2)] = np.einsum('ij,ik->jk', phi_sa_h1, phi_sa_h2)

    return {key: Omega_est_list[key] / len_batch_eval for key in Omega_est_list}


def compute_sigma_0_est(flattened_transitions, features_table_t, n_actions, len_batch_eval, H_bar):
    # Initialize Σ₀^est as a zero matrix of size Nc x Nc
    n_components = features_table_t.shape[1]
    Sigma_0_est = np.zeros((n_components, n_components))

    # Collect indices for each (s, a) pair
    indices = [
        (transition["state"][0] if isinstance(transition["state"], tuple) else transition["state"]) * n_actions +
        transition["action"]
        for transition in flattened_transitions
    ]

    # Gather the feature vectors for all transitions at once
    phi_sa_all = features_table_t[indices, :]

    # Use np.einsum to compute the sum of outer products
    # Sigma_0_est = np.einsum('ij,ik->jk', phi_sa_all, phi_sa_all)
    Sigma_0_est = np.dot(phi_sa_all.T, phi_sa_all)

    # Normalize by K_A2 * H_bar
    Sigma_0_est /= (len_batch_eval * H_bar)

    return Sigma_0_est


def compute_sigma_squared(batch_estimation, batch_evaluation, H, H_bar, n_components, n_valid_states, n_actions, gamma,
                          lambda_reg, target_policy, initial_state_distribution, Q_H, target_epsilon,features_table_t):
    """
    Compute sigma^2 using one batch for function estimation and another batch for evaluation.
    batch_estimation : A1. The batch of episodes used for function estimation (e.g., fitted Q-iteration).
    batch_evaluation : A2. The batch of episodes used for evaluating the estimated function and calculating sigma^2.
    Returns: sigma_squared : The estimated value of sigma^2 using the provided batches.
    """
    # Step 1: Estimate the function using batch_estimation (e.g., fitted Q-iteration)
    # q_table,_ = fitted_q_evaluation_valid_state(batch_estimation, n_valid_states, n_actions, H, gamma,lambda_reg, target_policy, Q_H,policy_epsilon)
    q_table, _ = fitted_q_evaluation_valid_state(batch_estimation, H, Q_H, target_policy, target_epsilon,
                                                 lambda_reg=0.001, goal_reward=-1, n_actions=4, n_valid_states=38)
    # fitted_q_evaluation_valid_state(batch_estimation, features_table_t, H, n_components, gamma, lambda_reg,optimal_policy)

    # unpack batch_evaluation
    flattened_transitions_batch_eval = list(chain.from_iterable(batch_evaluation))
    states_batch_eval = np.array(
        [t["state"][0] if isinstance(t["state"], tuple) else t["state"] for t in flattened_transitions_batch_eval])
    actions_batch_eval = np.array([t["action"] for t in flattened_transitions_batch_eval])
    next_states_batch_eval = np.array([t["next_state"] for t in flattened_transitions_batch_eval])
    rewards_batch_eval = np.array([t["reward"] for t in flattened_transitions_batch_eval])
    n_transition = len(flattened_transitions_batch_eval)

    feature_matrix_batch_eval = data_collected_embedded(flattened_transitions_batch_eval, features_table_t=np.eye(152))
    # Step 2: Compute nu_h and Sigma_0 using batch_evaluation
    # M_H_matrix = M_est_T_H_matrix(states_batch_eval, actions_batch_eval, next_states_batch_eval, H, features_table_t, target_policy, lambda_reg, Q_H,policy_epsilon)
    M_H_matrix = M_est_H_list(next_states_batch_eval, feature_matrix_batch_eval, n_transition, Q_H, target_epsilon,
                              lambda_reg,target_policy, H)
    # Compute nu_h for batch_evaluation
    # nu_h = compute_nu_h_est(H, M_H_matrix,initial_state_distribution,target_policy,Q_H[0],policy_epsilon,n_valid_states,features_table_t)
    nu_h_T = compute_nu_h_est_T(H, M_H_matrix, initial_state_distribution, target_policy, Q_H[0], target_epsilon,
                                n_valid_states, n_components)

    # Step 3: Compute Sigma0_est
    len_batch_eval = len(batch_evaluation)
    Sigma0_est = compute_sigma_0_est(flattened_transitions_batch_eval, features_table_t, n_actions, len_batch_eval,
                                     H_bar)  # Compute Sigma_0 for batch_evaluation
    Sigma0_est = Sigma0_est + lambda_reg * np.eye(152)

    # Step 4: Compute Omega using batch_evaluation
    epsilon = compute_epsilon(q_table, rewards_batch_eval, next_states_batch_eval, states_batch_eval,
                              actions_batch_eval, H, n_actions, target_policy, Q_H, target_epsilon)
    Omega_est_list = compute_Omega(flattened_transitions_batch_eval, features_table_t, epsilon, len_batch_eval, H,
                                   n_components)  # estimate_Omega(batch_evaluation, n_states, n_actions, epsilon,H)  # Compute Omega using batch_evaluation

    # Step 4: Calculate sigma_squared using the estimates from batch_estimation and batch_evaluation
    # Calculate term1: sum over h = 1 to H
    # term1 = sum((nu_h[h-1].T @ np.linalg.inv(Sigma0_est) @ Omega_est_list[(h, h)] @ np.linalg.inv(Sigma0_est) @ nu_h[h-1]) for h in range(H))
    term1 = sum(
        (nu_h_T[h] @ np.linalg.inv(Sigma0_est) @ Omega_est_list[(h, h)] @ np.linalg.inv(Sigma0_est) @ nu_h_T[h].T) for h
        in range(H))

    # Calculate term2: sum over h1 < h2
    # term2 = sum(2 * (nu_h[h1-1].T @ np.linalg.inv(Sigma0_est) @ Omega_est_list[(h1, h2)] @ np.linalg.inv(Sigma0_est) @ nu_h[h2-1]) for h1 in range(H) for h2 in range(h1 + 1, H))
    term2 = sum(2 * (
                nu_h_T[h1] @ np.linalg.inv(Sigma0_est) @ Omega_est_list[(h1, h2)] @ np.linalg.inv(Sigma0_est) @ nu_h_T[
            h2].T) for h1 in range(H) for h2 in range(h1 + 1, H))

    # Final sigma^2 estimate
    sigma_squared = (term1 + term2) / H_bar

    return sigma_squared


def subsampled_bootstrapping_fqe(trajectories, v_hat_full, Q_H, confidence_level, subset_size, B,
                                 initial_state_distribution, target_policy, target_epsilon,H,n_valid_states, n_actions):
    # rank_relaxation = 5
    # features_table_t = np.eye(152)
    # Step 1: Compute the FQE estimator for the full dataset
    # q_table, _ = fitted_q_evaluation_valid_state(trajectories, H, Q_H, target_policy,target_epsilon,lambda_reg = 0.001, goal_reward=-1, n_actions=4, n_valid_states=38)
    # v_hat_full, _ = value_return_estimation(q_table[0].reshape(n_valid_states, n_actions), initial_state_distribution, target_policy, target_epsilon)

    # Step 2: Initialize variables to store bootstrap results
    epsilon_list = []
    full_size = len(trajectories)

    # Step 3: Perform B bootstrap iterations
    for b in range(B):
        # print(b)
        # Build a random subset D^(b)_K,s
        subset_indices = np.random.choice(len(trajectories), size=subset_size, replace=False)
        D_b = [trajectories[i] for i in subset_indices]

        # flattened_Db = list(chain.from_iterable(D_b))
        # print(f"Subset length: {len(flattened_Db)}")
        # Sigma0_Db = compute_sigma_0_est(flattened_Db, features_table_t, n_actions, n_episodes, H_bar)
        # rank_Db = np.linalg.matrix_rank(Sigma0_Db)
        # print(f"Rank of Sigma0_Db: {rank_Db}")
        # if rank_Db < n_valid_states * n_actions - rank_relaxation:
        #    print("Skipping iteration due to singularity of D_b")
        #    continue

        # Generate a resample set D^(b)*_K,s by resampling from D_b
        resample_indices = np.random.choice(len(D_b), size=full_size, replace=True)
        D_b_star = [D_b[i] for i in resample_indices]

        # flattened_Db_star = list(chain.from_iterable(D_b_star))
        # Sigma0_Db_star = compute_sigma_0_est(flattened_Db_star, features_table_t, n_actions, n_episodes, H_bar)
        # rank_Db_star = np.linalg.matrix_rank(Sigma0_Db_star)
        # print(f"Rank of Sigma0_Db_star: {rank_Db_star}")
        # if rank_Db_star < n_valid_states * n_actions - rank_relaxation:
        #    print("Skipping iteration due to singularity of D_b_star")
        #    continue

        # Compute FQE estimator on the subset
        q_table_subset, _ = fitted_q_evaluation_valid_state(D_b, H, Q_H, target_policy, target_epsilon,
                                                            lambda_reg=0.001, goal_reward=-1, n_actions=4,
                                                            n_valid_states=38)
        v_hat_subset, _ = value_return_estimation(q_table_subset[0].reshape(n_valid_states, n_actions),
                                                  initial_state_distribution, target_policy, target_epsilon)

        # Compute FQE estimator on the resampled subset
        # v_hat_resample = fqe_estimator(D_b_star)
        q_table_resample, _ = fitted_q_evaluation_valid_state(D_b_star, H, Q_H, target_policy, target_epsilon,
                                                              lambda_reg=0.001, goal_reward=-1, n_actions=4,
                                                              n_valid_states=38)
        v_hat_resample, _ = value_return_estimation(q_table_resample[0].reshape(n_valid_states, n_actions),
                                                    initial_state_distribution, target_policy, target_epsilon)

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

