import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, RepetitionPenaltyLogitsProcessor

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

rollout_size = 0   # Number of tokens to generate during rollout
batch_size = 1     # Set to 1 for single text input
seed = 17          # Random seed for reproducibility

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
set_seed(seed)

# Load GPT-2 model and tokenizer
print("Loading GPT-2 model")
gpt = GPT2LMHeadModel.from_pretrained("gpt2")
gpt.eval()
gpt.to(device)
tokenizer_gpt = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer_gpt.padding_side = "left"
tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
eos_token_id = gpt.config.eos_token_id
vocab_size = tokenizer_gpt.vocab_size
print("GPT-2 model loaded")

# Utility functions for padding sequences
def pad_sequences_to_left(sequences, batch_first=False, padding_value=0):
    """Add left padding so sequences have the same shape."""
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        if batch_first:
            out_tensor[i, max_len-length:, ...] = tensor
        else:
            out_tensor[max_len-length:, i, ...] = tensor
    return out_tensor

def pad_sequences_to_left_states(sequences, padding_value=0, max_len=0):
    """Pad state tensors to have uniform dimensions across the batch."""
    max_size = sequences[0].size()
    out_dims = (max_size[0], max_size[1], len(sequences), max_size[2], max_len, max_size[4])
    out_tensor = sequences[0].new_full(out_dims, padding_value, device=device)
    for i, tensor in enumerate(sequences):
        length = tensor.size()[3]
        out_tensor[:, :, i, :, max_len-length:, ...] = tensor
    return out_tensor

# Functions for MCTS node evaluation
def root_fun(original_input, labels, temperature, repetition_penalty, get_discriminator_score):
    """Initialize root scores."""
    model_inputs = gpt.prepare_inputs_for_generation(original_input.input_ids, attention_mask=original_input.attention_mask, use_cache=True)
    with torch.no_grad():
        outputs = gpt(**model_inputs, return_dict=True)
        states = outputs.past_key_values

        prompt_masked_input_ids = torch.clone(model_inputs["input_ids"])
        inverted_attention_mask = model_inputs["attention_mask"] == 0
        prompt_masked_input_ids[inverted_attention_mask] = tokenizer_gpt.eos_token_id
        priors = repetition_penalty(prompt_masked_input_ids, outputs.logits[:, -1, :] / temperature)
        priors = F.softmax(priors, dim=-1).cpu().numpy()

    # Decode token IDs to text sequences
    texts = tokenizer_gpt.batch_decode(original_input.input_ids, skip_special_tokens=True)
    
    # Get values from the discriminator
    values = get_discriminator_score(texts).cpu().numpy()
    
    return priors, values, states

def rec_fun(states, token_ids, attention_masks, labels, temperature, repetition_penalty, rollout_size, get_discriminator_score):
    """Get scores from current nodes."""
    model_inputs = gpt.prepare_inputs_for_generation(token_ids, attention_mask=attention_masks, use_cache=True, past=states)
    with torch.no_grad():
        outputs = gpt(**model_inputs, return_dict=True)
        next_states = outputs.past_key_values

        prompt_masked_input_ids = torch.clone(token_ids)
        inverted_attention_mask = attention_masks == 0
        prompt_masked_input_ids[inverted_attention_mask] = tokenizer_gpt.eos_token_id

        priors = repetition_penalty(prompt_masked_input_ids, outputs.logits[:, -1, :] / temperature)
        priors = F.softmax(priors, dim=-1)

        # Optional rollout to simulate future steps
        if rollout_size > 0:
            for i in range(rollout_size):
                next_tokens = torch.unsqueeze(torch.argmax(priors, dim=-1), dim=1)
                token_ids = torch.cat((token_ids, next_tokens), dim=1)
                attention_masks = torch.cat((attention_masks, torch.ones_like(next_tokens, dtype=torch.long, device=device)), dim=1)
                prompt_masked_input_ids = torch.cat((prompt_masked_input_ids, next_tokens), dim=1)
                model_inputs = gpt.prepare_inputs_for_generation(token_ids, attention_mask=attention_masks, use_cache=True, past=outputs.past_key_values)
                with torch.no_grad():
                    outputs = gpt(**model_inputs, return_dict=True)
                    priors = repetition_penalty(prompt_masked_input_ids, outputs.logits[:, -1, :] / temperature)
                    priors = F.softmax(priors, dim=-1)

    # Decode token IDs to text sequences
    texts = tokenizer_gpt.batch_decode(token_ids, skip_special_tokens=True)
    
    # Get values from the discriminator
    values = get_discriminator_score(texts).cpu().numpy()
    
    return priors.cpu().numpy(), values, next_states

# Batched MCTS implementation (adjusted for batch_size=1)
class BatchedMCTS():
    def __init__(self, root_fun, rec_fun, batch_size, num_simulations, num_actions, num_sparse_actions, pb_c_init, temperature, alpha, penalty, rollout_size):
        # Initialize parameters
        self._batch_size = batch_size
        self._num_simulations = num_simulations
        self._num_actions = num_actions
        self._num_sparse_actions = min(num_sparse_actions, num_actions)
        self._pb_c_init = pb_c_init
        self._temperature = temperature
        self.alpha = alpha
        self.rollout_size = rollout_size

        self._root_fun = root_fun
        self._rec_fun = rec_fun
        self._adaptive_min_values = np.zeros(batch_size, dtype=np.float32)
        self._adaptive_max_values = np.zeros(batch_size, dtype=np.float32)
        self._labels = torch.zeros((batch_size,), dtype=torch.bool, device=device)

        # Allocate storage for tree structures
        num_nodes = num_simulations + 1
        batch_node = (batch_size, num_nodes)
        self._num_nodes = num_nodes
        self._visit_counts = np.zeros(batch_node, dtype=np.int32)
        self._values = np.zeros(batch_node, dtype=np.float32)
        self._likelihoods = np.zeros(batch_node, dtype=np.float32)
        self._raw_values = np.zeros(batch_node, dtype=np.float32)
        self._parents = np.zeros(batch_node, dtype=np.int32)
        self._action_from_parents = np.zeros(batch_node, dtype=np.int32)
        self._depth = np.zeros(batch_node, dtype=np.int32)
        self._is_terminal = np.full(batch_node, False, dtype=bool)

        batch_node_action = (batch_size, num_nodes, self._num_sparse_actions)
        self._topk_mapping = np.zeros(batch_node_action, dtype=np.int32)
        self._children_index = np.zeros(batch_node_action, dtype=np.int32)
        self._children_prior = np.zeros(batch_node_action, dtype=np.float32)
        self._children_values = np.zeros(batch_node_action, dtype=np.float32)
        self._children_visits = np.zeros(batch_node_action, dtype=np.int32)
        self._states = {}
        self._token_ids = {}
        self._attention_mask = {}
        self._batch_range = np.arange(batch_size)
        self._reset_tree()
        self._repetition_penalty = RepetitionPenaltyLogitsProcessor(penalty=penalty)

    def _reset_tree(self):
        """Resets the tree arrays."""
        self._visit_counts.fill(0)
        self._values.fill(0)
        self._likelihoods.fill(0)
        self._parents.fill(-1)
        self._action_from_parents.fill(-1)
        self._depth.fill(0)

        self._topk_mapping.fill(-1)
        self._children_index.fill(-1)
        self._children_prior.fill(0.0)
        self._children_values.fill(0.0)
        self._children_visits.fill(0)
        self._states = {}
        self._token_ids = {}
        self._attention_mask = {}

    def set_labels(self, labels):
        self._labels = labels

    def search(self, original_input):
        self._reset_tree()

        # Evaluate the root
        prior, values, states = self._root_fun(original_input, self._labels, self._temperature, self._repetition_penalty)
        self._adaptive_min_values = values
        self._adaptive_max_values = values + 1e-6

        root_index = 0
        self.create_node(root_index, prior, 1, values, states, original_input.input_ids, original_input.attention_mask, np.full(self._batch_size, False, dtype=bool))

        # Perform simulations
        leaf_indices = np.zeros((self._batch_size), np.int32)
        for sim in range(self._num_simulations):
            node_indices, actions = self.simulate()
            next_node_index = sim + 1  # Offset by 1 since root is 0
            self.expand(node_indices, actions, next_node_index)
            leaf_indices.fill(next_node_index)
            self.backward(leaf_indices)

        # Return the most visited tokens
        return self.dense_visit_counts()

    def dense_visit_counts(self):
        root_index = 0
        root_visit_counts = self._children_visits[:, root_index, :]
        dense_visit_counts = np.zeros((self._batch_size, self._num_actions))
        dense_visit_counts[self._batch_range[:, None], self._topk_mapping[:, root_index, :]] = root_visit_counts
        return dense_visit_counts

    def simulate(self):
        """Traverse the tree until reaching unexplored actions."""
        node_indices = np.zeros((self._batch_size), np.int32)
        while True:
            actions = self.uct_select_action(node_indices)
            next_node_indices = self._children_index[self._batch_range, node_indices, actions]
            is_unexplored = next_node_indices == -1
            if is_unexplored.all():
                return node_indices, actions
            else:
                node_indices = np.where(is_unexplored, node_indices, next_node_indices)

    def uct_select_action(self, node_indices):
        """Select actions using UCT algorithm."""
        node_children_prior = self._children_prior[self._batch_range, node_indices, :]
        node_children_values = self._children_values[self._batch_range, node_indices, :]
        node_children_visits = self._children_visits[self._batch_range, node_indices, :]
        node_visits = self._visit_counts[self._batch_range, node_indices]
        node_policy_score = np.sqrt(node_visits[:, None]) * self._pb_c_init * node_children_prior / (node_children_visits + 1)

        node_value_score = node_children_values
        node_uct_score = node_value_score + node_policy_score
        actions = np.argmax(node_uct_score, axis=1)
        return actions

    def get_states_from_node(self, b, n, d):
        """Reconstruct state tensors by traversing back to the root."""
        state_array = [None] * d
        state_array[d-1] = self._states[(b, n)]
        while n != 0:
            n = self._parents[(b, n)]
            d -= 1
            state_array[d-1] = self._states[(b, n)]
        result = torch.cat(state_array, 3)
        return result

    def expand(self, node_indices, actions, next_node_index):
        """Create and evaluate child nodes."""
        # Retrieve token ids for evaluation
        tokens_ids = pad_sequences_to_left([self._token_ids[(b, n)] for b, n in enumerate(node_indices)], True, eos_token_id)
        attention_masks = pad_sequences_to_left([self._attention_mask[(b, n)] for b, n in enumerate(node_indices)], True, 0)
        depths = torch.tensor([self._depth[(b, n)] + 1 for b, n in enumerate(node_indices)], device=device)
        children_priors = np.array([self._children_prior[(b, n)][actions[b]] for b, n in enumerate(node_indices)])
        likelihoods = np.array([self._likelihoods[(b, n)] for b, n in enumerate(node_indices)])
        previous_node_is_terminal = self._is_terminal[self._batch_range, node_indices[self._batch_range]]

        states_tensor = pad_sequences_to_left_states([self.get_states_from_node(b, n, depths[b].item()) for b, n in enumerate(node_indices)], 0, max_len=len(tokens_ids[0]))
        states = tuple(tuple(value for value in layer) for layer in states_tensor)

        # Convert sparse actions to dense actions
        dense_actions = self._topk_mapping[self._batch_range, node_indices, actions]
        tokens_ids = torch.cat((tokens_ids, torch.unsqueeze(torch.tensor(dense_actions, device=device), 1)), dim=1)
        attention_masks = torch.cat((attention_masks, torch.ones(len(dense_actions), 1, dtype=torch.long, device=device)), dim=1)

        # Check for terminal nodes
        expanded_node_is_terminal = dense_actions == eos_token_id

        # Evaluate new nodes
        prior, values, next_states = self._rec_fun(states, tokens_ids, attention_masks, self._labels, self._temperature, self._repetition_penalty, self.rollout_size)
        self.create_node(next_node_index, prior, likelihoods * children_priors, values, next_states, tokens_ids, attention_masks, expanded_node_is_terminal)

        # Update min and max values
        self._adaptive_min_values = np.minimum(self._adaptive_min_values, values)
        self._adaptive_max_values = np.maximum(self._adaptive_max_values, values)

        # Update tree topology
        self._children_index[self._batch_range, node_indices, actions] = next_node_index
        self._parents[:, next_node_index] = node_indices
        self._action_from_parents[:, next_node_index] = actions
        self._depth[:, next_node_index] = self._depth[self._batch_range, node_indices] + 1

    def create_node(self, node_index, prior, likelihoods, values, next_states, tokens_ids, attention_masks, expanded_node_is_terminal):
        """Create nodes with computed values."""
        prior_topk_indices = np.argpartition(prior, -self._num_sparse_actions, axis=-1)[:, -self._num_sparse_actions:]
        prior = prior[self._batch_range[:, None], prior_topk_indices]

        self._topk_mapping[self._batch_range, node_index, :] = prior_topk_indices
        self._children_prior[:, node_index, :] = prior
        self._likelihoods[:, node_index] = likelihoods

        raw_values = values ** self.alpha * likelihoods ** (1 - self.alpha)
        self._values[:, node_index] = raw_values
        self._raw_values[:, node_index] = raw_values
        self._visit_counts[:, node_index] = 1
        self._is_terminal[:, node_index] = expanded_node_is_terminal

        # Transform states for easier manipulation
        key_value_tensor = torch.stack([torch.stack(list(next_states[i]), dim=0) for i in range(len(next_states))], dim=0)
        if node_index == 0:
            for b in range(len(tokens_ids)):
                self._states[(b, node_index)] = key_value_tensor[:, :, b]
        else:
            for b in range(len(tokens_ids)):
                self._states[(b, node_index)] = key_value_tensor[:, :, b, :, -1:]

        # Update tokens and attention masks
        for b, token_ids in enumerate(tokens_ids):
            self._token_ids[(b, node_index)] = token_ids
        for b, attention_mask in enumerate(attention_masks):
            self._attention_mask[(b, node_index)] = attention_mask

    def backward(self, leaf_indices):
        """Backpropagate values up the tree."""
        node_indices = leaf_indices
        leaf_values = self._values[self._batch_range, leaf_indices]
        while True:
            is_root = node_indices == 0
            if is_root.all():
                break
            parents = np.where(is_root, 0, self._parents[self._batch_range, node_indices])
            not_root_mask_int = (1 - is_root)
            not_root_mask = 1.0 - is_root.astype(float)
            self._values[self._batch_range, parents] = not_root_mask * (
                self._values[self._batch_range, parents] * self._visit_counts[self._batch_range, parents] + leaf_values
            ) / (self._visit_counts[self._batch_range, parents] + 1.0) + is_root.astype(float) * self._values[self._batch_range, parents]

            self._visit_counts[self._batch_range, parents] += not_root_mask_int
            actions = np.where(is_root, 0, self._action_from_parents[self._batch_range, node_indices])
            self._children_values[self._batch_range, parents, actions] = not_root_mask * self._values[self._batch_range, node_indices] + is_root.astype(float) * self._children_values[self._batch_range, parents, actions]
            self._children_visits[self._batch_range, parents, actions] += not_root_mask_int
            node_indices = parents

def main(c, alpha, temperature, penalty, num_it, get_discriminator_score, prompt_text):
    labels = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    # If labels are not used, you can skip setting them

    repetition_penalty = RepetitionPenaltyLogitsProcessor(penalty=penalty)

    MCTS = BatchedMCTS(
        lambda original_input, labels, temp, rep_penalty: root_fun(
            original_input, labels, temp, repetition_penalty, get_discriminator_score
        ),
        lambda states, token_ids, attention_masks, labels, temp, rep_penalty, roll_size: rec_fun(
            states, token_ids, attention_masks, labels, temp, repetition_penalty, roll_size, get_discriminator_score
        ),
        batch_size=batch_size,
        num_simulations=num_it,
        num_actions=vocab_size,
        num_sparse_actions=50,
        pb_c_init=c,
        temperature=temperature,
        alpha=alpha,
        penalty=penalty,
        rollout_size=rollout_size,
    )
    MCTS.set_labels(labels)
    original_input = tokenizer_gpt(prompt_text, return_tensors="pt", padding=True, add_special_tokens=False, max_length=15, truncation=True).to(device)
    generated_text = prompt_text
    num_tokens_to_generate = 50  # Adjust the number of tokens to generate
    for _ in tqdm(range(num_tokens_to_generate), desc="Generating text"):
        res_search = MCTS.search(original_input)
        next_token_id = np.argmax(res_search, axis=1)[0]
        next_token = tokenizer_gpt.decode([next_token_id])
        generated_text += next_token
        
        # Update original_input with the new token
        original_input.input_ids = torch.cat((original_input.input_ids, torch.tensor([[next_token_id]], device=device)), dim=1)
        original_input.attention_mask = torch.cat((original_input.attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)), dim=1)

    print("Generated text:")
    print(generated_text)

if __name__ == "__main__":
    main()