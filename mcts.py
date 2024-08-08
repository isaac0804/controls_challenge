import numpy as np
import torch

from muzeronet import MuZeroNet
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

class Node:
    def __init__(self, prior):
        self.visit_count = 0 # N
        self.prior = prior # P
        self.value_sum = 0 # Q
        self.children = {}
        self.hidden_state = None # S
        self.reward = 0 # R

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

def ucb_score(parent, child, min_q, max_q):
    pb_c = 1.25
    pb_c_init = 1.25
    pb_c_base = 19652
    c = np.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = c * child.prior
    value_score = child.value()
    if max_q != min_q:
        value_score = min_q + (value_score - min_q) / (max_q - min_q)
    return prior_score + value_score

def select_action(model, state, num_simulations=50):
    root = Node(0)
    root.hidden_state, policy_logits, value = model.initial_inference(state)
    # TODO: Scale back reward and value
    expand_node(root, policy_logits)

    for _ in range(num_simulations):
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(node, explore=False)
            search_path.append(node)

        parent = search_path[-2]
        hidden_state, reward, policy_logits, value = model.recurrent_inference(parent.hidden_state, torch.tensor(action).unsqueeze(0))
        node.hidden_state = hidden_state
        node.reward = reward
        expand_node(node, policy_logits)
        backpropagate(search_path, value, model.discount)

        min_q, max_q = min(child.value() for child in root.children.values()), max(child.value() for child in root.children.values())
        if min_q != max_q:
            for child in root.children.values():
                child.value_sum = min_q + (child.value_sum - min_q) / (max_q - min_q)

        # print([child.visit_count for child in root.children.values()])
        # print([child.value_sum.item() if isinstance(child.value_sum,torch.Tensor) else child.value_sum for child in root.children.values()])
        # print(np.array([child.value_sum.item() if isinstance(child.value_sum,torch.Tensor) else child.value_sum for child in root.children.values()]))
        # print([child.value() for child in root.children.values()])

    # print([child.visit_count for child in root.children.values()])
    # print("--------------------------------------")
    return select_child(root)[0]

def expand_node(node, policy_logits):
    policy = torch.softmax(policy_logits, dim=1).squeeze()
    for action, p in enumerate(policy):
        node.children[action] = Node(p.item())

def select_child(node, explore=False):
    # Return: Tuple (action index, node)
    min_q, max_q = min(child.value() for child in node.children.values()), max(child.value() for child in node.children.values())
    if explore:
        p = [ucb_score(node, item[1], min_q, max_q) for item in node.children.items()]
        p = torch.nn.functional.softmax(torch.tensor(p, dtype=torch.float64),dim=-1)
        return list(node.children.items())[np.random.choice(np.arange(0,21), p=p)]
    else:
        return max(node.children.items(), key=lambda item: ucb_score(node, item[1], min_q, max_q))

def backpropagate(search_path, value, discount):
    for node in reversed(search_path):
        # node.value_sum += value
        node.value_sum = (node.visit_count*node.value_sum + value)/(node.visit_count+1)
        node.visit_count += 1
        value = node.reward + discount * value

# Usage example:
# model = MuZeroNet(5, 21)
# state = torch.tensor([[-0.4228678 , 32.58351451,  0.04527782, -0.6419921 ,  1.3655914 ]])
# action = select_action(model, state, 100)
# print(action)