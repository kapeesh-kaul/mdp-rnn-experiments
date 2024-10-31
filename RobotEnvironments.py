import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch

class RobotGridMDP:
    def __init__(self, gamma=0.8, epsilon=1e-3):
        self.gamma = gamma
        self.epsilon = epsilon
        self.states = {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0, 's6': 0}
        # self.rewards = {
        #     ('s2', 's3'): 50,
        #     ('s6', 's3'): 100,
        # }
        self.transitions = {
            's1': [('s2', 0), ('s4', 0)],
            's2': [('s1', 0), ('s5', 0), ('s3', 50)],
            's3': [('s3', 0)],  # self-loop
            's4': [('s1', 0), ('s5', 0)],
            's5': [('s2', 0), ('s4', 0), ('s6', 0)],
            's6': [('s5', 0), ('s3', 100)]
        }
    
    def plot_state_values(self, values):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(6, 4))

        state_labels = list(values.keys())
        state_values = [values[state] for state in state_labels]

        for i, (label, value) in enumerate(zip(state_labels, state_values)):
            row, col = divmod(i, 3)
            ax[row, col].text(0.5, 0.5, f'{label} : {value:.1f}', fontsize=14, ha='center')
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])

        plt.tight_layout()
        plt.show()

    def perform_value_iteration(self):
        iterations = 0
        converged = False
        while not converged:
            new_values = self.states.copy()
            max_delta = 0
            for state in self.states:
                max_value = float('-inf')
                for (next_state, reward) in self.transitions[state]:
                    value = reward + self.gamma * self.states[next_state]
                    max_value = max(max_value, value)
                new_values[state] = max_value
                max_delta = max(max_delta, abs(new_values[state] - self.states[state]))
            self.states = new_values
            iterations += 1
            if max_delta < self.epsilon:
                converged = True
        return self.states, iterations

    def extract_optimal_policy(self):
        policy = {}
        for state in self.states:
            best_action = None
            best_value = float('-inf')
            for (next_state, reward) in sorted(self.transitions[state], key=lambda x: x[0]):
                value = reward + self.gamma * self.states[next_state]
                if value > best_value:
                    best_value = value
                    best_action = next_state
            policy[state] = best_action
        return policy

    def plot_policy(self, policy):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 2)

        # Draw grid lines
        for x in range(4):
            ax.axvline(x, color='gray', lw=1)
        for y in range(3):
            ax.axhline(y, color='gray', lw=1)
        
        # Define coordinates for each state
        state_positions = {
            's1': (0.5, 1.5), 's2': (1.5, 1.5), 's3': (2.5, 1.5),
            's4': (0.5, 0.5), 's5': (1.5, 0.5), 's6': (2.5, 0.5)
        }

        for state, action in policy.items():
            start_pos = state_positions[state]
            if action == state:
                # Self-loop with 3/4 circle using Arc
                arc = Arc((start_pos[0], start_pos[1]), 0.3, 0.3, angle=0, theta1=45, theta2=315, color='blue', lw=1.5)
                ax.add_patch(arc)
                # Adding an arrowhead at the end of the arc for the self-loop
                ax.annotate('', xy=(start_pos[0] + 0.15, start_pos[1] - 0.15), 
                            xytext=(start_pos[0] + 0.1, start_pos[1] - 0.1),
                            arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->'))
            else:
                end_pos = state_positions[action]
                
                # Calculate direction and shorten arrow length by adding padding on both sides
                dx = (end_pos[0] - start_pos[0]) * 0.6  # Scale down for shorter arrow
                dy = (end_pos[1] - start_pos[1]) * 0.6

                # Offset start position slightly towards the end position
                start_x = start_pos[0] + (dx * 0.3)
                start_y = start_pos[1] + (dy * 0.3)

                # Draw arrow with reduced length
                ax.arrow(start_x, start_y, dx, dy, head_width=0.05, head_length=0.05, fc='blue', ec='blue')

        # Annotate states
        for state, pos in state_positions.items():
            ax.text(pos[0], pos[1], state, ha='center', va='center', fontsize=12, color="black", fontweight="bold")

        ax.set_aspect('equal')
        ax.axis('off')
        plt.title("Optimal Policy with Arrows Indicating Routes and Self-Loops")
        plt.show()

    def plot_environment(self):
        """
        Plots each state transition with arrows indicating direction and labels showing rewards.
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 2)

        # Draw grid lines
        for x in range(4):
            ax.axvline(x, color='gray', lw=1)
        for y in range(3):
            ax.axhline(y, color='gray', lw=1)
        
        # Define coordinates for each state
        state_positions = {
            's1': (0.5, 1.5), 's2': (1.5, 1.5), 's3': (2.5, 1.5),
            's4': (0.5, 0.5), 's5': (1.5, 0.5), 's6': (2.5, 0.5)
        }

        # Plot each transition with rewards
        for state, transitions in self.transitions.items():
            start_pos = state_positions[state]
            for (next_state, reward) in transitions:
                end_pos = state_positions[next_state]
                if next_state == state:
                    # Self-loop for transitions where next state is the same as the current state
                    arc = Arc((start_pos[0], start_pos[1]), 0.3, 0.3, angle=0, theta1=45, theta2=315, color='blue')
                    ax.add_patch(arc)
                    ax.text(start_pos[0] + 0.25, start_pos[1] - 0.25, f"{reward}", color='red', fontsize=10)
                else:
                    # Draw arrow between different states
                    dx = (end_pos[0] - start_pos[0]) * 0.6  # Scale down for shorter arrow
                    dy = (end_pos[1] - start_pos[1]) * 0.6

                    # Offset start position slightly to create space for text
                    start_x = start_pos[0] + dx * 0.2
                    start_y = start_pos[1] + dy * 0.2

                    # Draw arrow with reduced length and small arrowhead
                    ax.arrow(start_x, start_y, dx, dy, head_width=0.05, head_length=0.05, fc='blue', ec='blue')

                    # Place reward text in the middle of the arrow with offset for visibility
                    mid_x = start_pos[0] + dx * 0.5 + 0.05
                    mid_y = start_pos[1] + dy * 0.5 + 0.05
                    ax.text(mid_x, mid_y, f"{reward}", color='red', fontsize=10)

        # Annotate states
        for state, pos in state_positions.items():
            ax.text(pos[0], pos[1], state, ha='center', va='center', fontsize=12, color="black", fontweight="bold")

        ax.set_aspect('equal')
        ax.axis('off')
        plt.title("State Transitions with Rewards")
        plt.show()

class RobotGridMDPWithIce:
    def __init__(self, gamma=0.8, epsilon=1e-3):
        self.gamma = gamma
        self.epsilon = epsilon
        self.states = {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0, 's6': 0}
        # self.rewards = {
        #     ('s2', 's3'): 50,
        #     ('s6', 's3'): 100,
        # }
        self.transitions = {
            's1': [('s2', 0), ('s4', 0)],
            's2': [('s1', 0), ('s5', 0), ('s3', 50)],
            's3': [('s3', 0)],  # self-loop
            's4': [('s1', 0), ('s5', 0)],
            's5': [('s2', 0), ('s4', 0), ('s6', 0)],
            's6': [('s5', 0), ('s3', 100, 'ice'), ('s6', 0, 'ice')]  # Non-deterministic transitions due to ice
        }

    def perform_value_iteration(self, p=1.0):
        """
        Perform value iteration with a non-deterministic transition for s6 -> s3 with probability p.
        :param p: Probability of successful transition from s6 to s3
        :return: A dictionary of optimal values for each state and the number of iterations taken.
        """
        iterations = 0
        converged = False
        while not converged:
            new_values = self.states.copy()
            max_delta = 0
            for state in self.states:
                max_value = float('-inf')
                for (next_state, reward, *conditions) in self.transitions[state]:
                    if conditions == ['ice'] and state == 's6':
                        # Non-deterministic transition for s6 -> s3 with probability p
                        value = p * (reward + self.gamma * self.states['s3']) + (1 - p) * (0 + self.gamma * self.states['s6'])
                    else:
                        # Deterministic transitions
                        value = reward + self.gamma * self.states[next_state]
                    max_value = max(max_value, value)
                new_values[state] = max_value
                max_delta = max(max_delta, abs(new_values[state] - self.states[state]))
            self.states = new_values
            iterations += 1
            if max_delta < self.epsilon:
                converged = True
        return self.states, iterations

    def extract_optimal_policy(self, p=1.0):
        """
        Determine the optimal policy by selecting the action with the maximum value for each state, given probability p.
        :param p: Probability of successful transition from s6 to s3
        :return: A dictionary representing the optimal action for each state.
        """
        policy = {}
        for state in self.states:
            best_action = None
            best_value = float('-inf')
            for (next_state, reward, *conditions) in sorted(self.transitions[state], key=lambda x: x[0]):
                if conditions == ['ice'] and state == 's6':
                    # Non-deterministic transition for s6 -> s3 with probability p
                    value = p * (reward + self.gamma * self.states['s3']) + (1 - p) * (0 + self.gamma * self.states['s6'])
                else:
                    # Deterministic transitions
                    value = reward + self.gamma * self.states[next_state]
                if value > best_value:
                    best_value = value
                    best_action = next_state
            policy[state] = best_action
        return policy

    def find_threshold_p(self):
        """
        Find the threshold probability p at which the optimal policy for s6 avoids the ice patch.
        :return: Threshold probability p below which the policy avoids the ice.
        """
        low, high = 0.0, 1.0  # Start with p from 0 to 1
        threshold_p = 0.0
        while high - low > 1e-3:
            mid_p = (low + high) / 2
            self.perform_value_iteration(p=mid_p)
            policy = self.extract_optimal_policy(p=mid_p)
            if policy['s6'] == 's5':  # The robot prefers "go west"
                threshold_p = mid_p
                high = mid_p
            else:
                low = mid_p
        return threshold_p
    
    def plot_policy(self, policy):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 2)

        # Draw grid lines
        for x in range(4):
            ax.axvline(x, color='gray', lw=1)
        for y in range(3):
            ax.axhline(y, color='gray', lw=1)
        
        # Define coordinates for each state
        state_positions = {
            's1': (0.5, 1.5), 's2': (1.5, 1.5), 's3': (2.5, 1.5),
            's4': (0.5, 0.5), 's5': (1.5, 0.5), 's6': (2.5, 0.5)
        }

        for state, action in policy.items():
            start_pos = state_positions[state]
            if action == state:
                # Self-loop with 3/4 circle using Arc
                arc = Arc((start_pos[0], start_pos[1]), 0.3, 0.3, angle=0, theta1=45, theta2=315, color='blue', lw=1.5)
                ax.add_patch(arc)
                # Adding an arrowhead at the end of the arc for the self-loop
                ax.annotate('', xy=(start_pos[0] + 0.15, start_pos[1] - 0.15), 
                            xytext=(start_pos[0] + 0.1, start_pos[1] - 0.1),
                            arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->'))
            else:
                end_pos = state_positions[action]
                
                # Calculate direction and shorten arrow length by adding padding on both sides
                dx = (end_pos[0] - start_pos[0]) * 0.6  # Scale down for shorter arrow
                dy = (end_pos[1] - start_pos[1]) * 0.6

                # Offset start position slightly towards the end position
                start_x = start_pos[0] + (dx * 0.3)
                start_y = start_pos[1] + (dy * 0.3)

                # Draw arrow with reduced length
                ax.arrow(start_x, start_y, dx, dy, head_width=0.05, head_length=0.05, fc='blue', ec='blue')

        # Annotate states
        for state, pos in state_positions.items():
            ax.text(pos[0], pos[1], state, ha='center', va='center', fontsize=12, color="black", fontweight="bold")

        ax.set_aspect('equal')
        ax.axis('off')
        plt.title("Optimal Policy with Arrows Indicating Routes and Self-Loops")
        plt.show()
