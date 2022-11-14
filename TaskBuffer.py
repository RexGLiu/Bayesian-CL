import numpy as np


class TaskBuffer:
    def __init__(self, n_tasks, max_set_size, x_dim, y_dim):
        self.n_tasks = n_tasks
        self.max_set_size = max_set_size

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.task_pos = 0
        self.full = False

        # replay buffer organised by task index followed by a data index for each task
        self.x = np.zeros((self.n_tasks, self.max_set_size, self.x_dim), dtype=np.float32)
        self.y = np.zeros((self.n_tasks, self.max_set_size, self.y_dim), dtype=np.float32)
        self.set_size = np.zeros(self.n_tasks, dtype=np.int_)

    def add(self, x_set, y_set, set_size):
        # adds new dataset to mem buffer

        self.x[self.task_pos, :set_size] = x_set
        self.y[self.task_pos, :set_size] = y_set
        self.set_size[self.task_pos] = set_size

        if self.task_pos + 1 == self.n_tasks:  # buffer full
            self.full = True
        self.task_pos = (self.task_pos + 1) % self.n_tasks

    def sample(self, batch_size=1):
        if self.full:
            indices = np.random.choice(self.n_tasks, batch_size)
        else:
            indices = np.random.choice(self.pos, batch_size)

        return self.x[indices], self.y[indices], self.set_size[indices]