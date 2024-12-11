import torch
import torch.nn as nn
import torch.utils.data

import numpy as np
import scipy
import json
import scipy.spatial
import scipy.spatial.distance


def load_data_from_json(filename):
    with open(filename, 'r') as f:
        task_sols = json.load(f)
        
    X = []
    Y = []
   

    for task_sol in task_sols["task_sols"]:
        X.append(task_sol["condensed_task"])
        Y.append(task_sol["solution"])
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

def compute_diff(X_data, Y_data, r, move_r_mult=10**(1/18)):
    x_diff = np.ndarray(X_data.shape[0])
    y_diff = np.ndarray(X_data.shape[0])

    tree = scipy.spatial.KDTree(X_data)
    indices = tree.query_ball_point(X_data, r * move_r_mult, workers=-1)
    for i, index in enumerate(indices):
        x_diff[i] = np.linalg.norm(X_data[index].mean(0) - X_data[i])
        y_diff[i] = np.linalg.norm(Y_data[index].mean(0) - Y_data[i])

    return np.array(x_diff).flatten(), np.array(y_diff).flatten()

def filter_data_dist(X_data, Y_data, r):
    cum = 1
    for i in range(30, 18, -1):
        x_diff, y_diff = compute_diff(X_data, Y_data, r)
        condition = y_diff < i
        X_data, Y_data = X_data[condition], Y_data[condition]
        cum *= condition.mean()
        print(i, cum, condition.mean())
    return X_data, Y_data

def get_kl_weight(epoch, start, end, max_weight):
    if epoch < start:
        return 0.0
    elif epoch > end:
        return max_weight
    else:
        return max_weight * (epoch - start) / (end - start)
    
def traj_smoothness_loss(traj, labda_smooth = 0.001, joint_size=6):
    """
    traj: [batch, joint_size*time_horizon]
    """
    # TODO: need verification
    traj = traj.view(-1, joint_size, -1)
    traj = traj.permute(0, 2, 1)
    traj_diff = traj[:, 1:] - traj[:, :-1]
    traj_diff = traj_diff.pow(2).sum(-1).mean()
    return labda_smooth * traj_diff
    


class JointDataset(torch.utils.data.Dataset):
    def __init__(self, x_filename, y_filename, device, allow_sub_sample=False, sub_sample_size=None):
        with open(x_filename, "rb") as f:
            self.X = torch.load(f).to(device)
        with open(y_filename, "rb") as f:
            self.Y = torch.load(f).to(device)
            
        # Convert to float32
        self.X = self.X.float()
        self.Y = self.Y.float()

        assert (self.X.shape[0] == self.Y.shape[0])

        if allow_sub_sample and sub_sample_size is not None:
            indices = np.random.choice(self.X.shape[0], sub_sample_size, replace=False)
            self.X = self.X[indices]
            self.Y = self.Y[indices]

        self.X_normalise_params = [self.X.mean(dim=0), self.X.std(dim=0)]
        self.Y_normalise_params = [self.Y.mean(dim=0), self.Y.std(dim=0)]

        self.rows = self.X.shape[0]
        self.input_size = self.X.shape[1]
        self.output_size = self.Y.shape[1]

    def __len__(self):
        return self.rows

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    
class ARJointDataset(torch.utils.data.Dataset):
    def __init__(self, x_filename, y_filename, device, allow_sub_sample=False, sub_sample_size=None):
        with open(x_filename, "rb") as f:
            self.X = torch.load(f).to(device)
        with open(y_filename, "rb") as f:
            Y = torch.load(f).to(device)
            self.Y = Y.reshape(-1, 120, 6) #[sample, time_horrizon, joint]
            
        # Convert to float32
        self.X = self.X.float()
        self.Y = self.Y.float()

        if allow_sub_sample and sub_sample_size is not None:
            indices = np.random.choice(self.X.shape[0], sub_sample_size, replace=False)
            self.X = self.X[indices]
            self.Y = self.Y[indices]

        # Check if X and Y have the same number of samples
        assert (self.X.shape[0] == self.Y.shape[0])
        
        self.input_size = self.X.shape[1]
        self.output_size = 6

        # Process the data to create individual states
        self.processed_data = []
        self._process_data()

    def _process_data(self):
        for i in range(self.Y.shape[0]):
            trajectory = self.Y[i]  # Trajectory of states for a single sample
            condition = self.X[i, :-6]  # Condition part of X (excluding last 6 dimensions)

            for t in range(trajectory.shape[0] - 1):
                current_state = trajectory[t]
                next_state = trajectory[t + 1]

                processed_x = torch.cat((condition, current_state), dim=0)
                processed_y = next_state - current_state # Predict the change in state
                t = torch.tensor([t], dtype=torch.float32).to(self.X.device)

                self.processed_data.append((processed_x, processed_y, t))

        self.rows = len(self.processed_data)

    def __len__(self):
        return self.rows

    def __getitem__(self, idx):
        return self.processed_data[idx]


class LSTMJointDataset(torch.utils.data.Dataset):
    TIME_LENGTH = 120

    def __init__(self, x_filename, y_filename, device, output):
        with open(x_filename, "rb") as f:
            self.X = torch.load(f).to(device)
            self.X = self.X[:, None, :].expand(-1, self.TIME_LENGTH, -1)
            times = torch.tensor(list(range(self.TIME_LENGTH))) / self.TIME_LENGTH
            positional_encoding = times[None, :, None].expand(self.X.shape[0], -1, 1).to(device)
            self.X = torch.cat((self.X, positional_encoding), dim=-1)
        with open(y_filename, "rb") as f:
            self.Y = torch.load(f).to(device)
            self.Y = self.Y.view(self.Y.shape[0], -1, output)

        assert (self.X.shape[0] == self.Y.shape[0])

        self.X_normalise_params = [self.X.mean(dim=[0, 1]), self.X.std(dim=[0, 1])]
        self.Y_normalise_params = [self.Y.mean(dim=[0, 1]), self.Y.std(dim=[0, 1])]
        self.rows = self.X.shape[0]

    def __len__(self):
        return self.rows

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


