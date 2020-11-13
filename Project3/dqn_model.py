import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


def save(model, path):
    torch.save(model.state_dict(), path)


def load(path, n_actions):
    model = DuelingDQN(n_actions=n_actions)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


class DQN(nn.Module):
    def __init__(self, n_actions, in_channels=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.lin = nn.Linear(7 * 7 * 64, 512)
        self.last = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.lin(x.view(x.size(0), -1)))
        return self.last(x)


class DuelingDQN(nn.Module):
    def __init__(self, n_actions, in_channels=4):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        nn.init.constant_(self.conv3.bias, 0)

        self.conv4 = nn.Conv2d(64, 1024, kernel_size=7, stride=1)
        nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
        nn.init.constant_(self.conv4.bias, 0)

        self.fc_value = nn.Linear(512, 1)
        nn.init.kaiming_normal_(self.fc_value.weight, nonlinearity='relu')
        self.fc_advantage = nn.Linear(512, n_actions)
        nn.init.kaiming_normal_(self.fc_advantage.weight, nonlinearity='relu')

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x_value = x[:, :512, :, :].view(-1, 512)
        x_advantage = x[:, 512:, :, :].view(-1, 512)
        x_value = self.fc_value(x_value)
        x_advantage = self.fc_advantage(x_advantage)
        q_value = x_value + x_advantage.sub(torch.mean(x_advantage, 1)[:, None])
        return q_value
