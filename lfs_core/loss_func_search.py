from lfs_core.utils.loss import loss_search
from lfs_core.utils.loss_agent import LFSAgent
import lfs_core.link_utils as link
import torch
import torch.distributed as dist


class LossFuncSearch(object):
    def __init__(self, sm=32):
        self.model = None

        self.lr = 0.05
        self.sample_step = 2
        self.val_freq = 2
        self.scale = 0.2
        self.global_rank = dist.get_rank()
        self.best_acc = 0
        self.best_epoch = -1
        self.sm = sm
        self.__init_agent()

    def __init_agent(self):
        self.agent = LFSAgent(self.lr, self.scale)
        self.p = [i / 10.0 for i in range(11)]
        self.a = [-1.0] + [0.0, ] * 9

    def set_model(self, model):
        self.model = model

    def get_loss(self, outputs, targets):
        loss = loss_search(outputs, targets, self.p, self.a, self.sm)
        return loss

    def set_loss_parameters(self, epoch):
        if epoch >= 2:
            self.p, self.a = self.agent.sample_subfunction()

    def _broadcast_parameters(self, rank):
        """
        broadcast model parameters 
        """
        link.broadcast_params(self.model, rank)

    def update_lfs(self, reward):
        rank = self.global_rank
        temp_acc = torch.tensor(reward)

        test_acc_tensor = link.all_gather(temp_acc)
        best_test_acc_rank = torch.argmax(test_acc_tensor)
        current_best_acc = test_acc_tensor[best_test_acc_rank].item()

        self._broadcast_parameters(rank=best_test_acc_rank.item())

        reward = (test_acc_tensor - torch.mean(test_acc_tensor)) / ((torch.max(test_acc_tensor) - torch.min(test_acc_tensor)) + 1e-6) * 2

        self.agent.step(reward=reward[rank].item())

        return current_best_acc
