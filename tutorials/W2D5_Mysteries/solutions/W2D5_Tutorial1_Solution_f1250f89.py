class SharedWorkspace(nn.Module):

    def __init__(self, num_specialists, hidden_dim, num_memory_slots, memory_slot_dim):
        super().__init__()
        self.num_specialists = num_specialists
        self.hidden_dim = hidden_dim
        self.num_memory_slots = num_memory_slots
        self.memory_slot_dim = memory_slot_dim
        self.workspace_memory = nn.Parameter(torch.randn(num_memory_slots, memory_slot_dim))

        # Attention mechanism components for writing to the workspace
        self.key = nn.Linear(hidden_dim, memory_slot_dim)
        self.query = nn.Linear(memory_slot_dim, memory_slot_dim)
        self.value = nn.Linear(hidden_dim, memory_slot_dim)

    def write_to_workspace(self, specialists_states):
        # Flatten specialists' states if they're not already
        specialists_states = specialists_states.view(-1, self.hidden_dim)

        # Compute key, query, and value
        keys = self.key(specialists_states)
        query = self.query(self.workspace_memory)
        values = self.value(specialists_states)

        # Compute attention scores and apply softmax
        attention_scores = torch.matmul(query, keys.transpose(-2, -1)) / (self.memory_slot_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Update workspace memory with weighted sum of values
        updated_memory = torch.matmul(attention_probs, values)
        self.workspace_memory = nn.Parameter(updated_memory)

        return self.workspace_memory

    def forward(self, specialists_states):
        updated_memory = self.write_to_workspace(specialists_states)
        return updated_memory