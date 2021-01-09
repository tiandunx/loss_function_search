import torch
import lfs_core.link_utils as link


class LFSAgent(object):
    def __init__(self, lr=1e-4, scale=0.1):

        self.counter = 0
        self.log_prob = []
        self.actions = []
        self.gaussian_param_loc = torch.nn.Parameter(torch.Tensor([0.0, ] * 10))
        self.gaussian_scale = torch.Tensor([scale, ] * 10)
        self.gaussian_optimizer = torch.optim.Adam([self.gaussian_param_loc], lr=lr, betas=(0.5, 0.999), weight_decay=0.0)
        self.relu = torch.nn.ReLU()

    def sample_subfunction(self):
        p, a = self.gaussian_sample_subfunction()
        return p, a

    def step(self, reward=0.0):
        self.gaussian_step(reward)

    def gaussian_sample_subfunction(self):
        p = [i / 10.0 for i in range(11)]
        a = []
        m = [torch.distributions.normal.Normal(self.relu(self.gaussian_param_loc[0]), self.gaussian_scale[0])]
        for i in range(1, 10):
            m.append(torch.distributions.normal.Normal(self.gaussian_param_loc[i], self.gaussian_scale[i]))

        for i in range(10):
            x = m[i].sample().item()
            x = -abs(x)
            a.append(x)
        self.actions.append(torch.tensor(a))

        return p, a

    def add_multi_gaussian_log_prob(self, actions):
        # IMPORTANT: the main function for the agent in the main process compute log probability of act1 and act2
        # if cuda:
        self.gaussian_loc_param_cuda = self.gaussian_param_loc.cuda()
        self.gaussian_scale_cuda = self.gaussian_scale.cuda()

        actions_ = torch.stack(actions, dim=1).cuda()

        m = [torch.distributions.normal.Normal(self.relu(self.gaussian_loc_param_cuda[0]), self.gaussian_scale_cuda[0])]
        for i in range(1, 10):
            m.append(torch.distributions.normal.Normal(self.gaussian_loc_param_cuda[i], self.gaussian_scale_cuda[i]))

        for i in range(10):
            self.log_prob.append(torch.sum(m[i].log_prob(actions_[i])))

    def scale_step(self, epoch, tot_epoch=1000, start_scale=0.1, final_scale=0.01):
        temp_scale = start_scale + (final_scale - start_scale) * (epoch / tot_epoch)
        self.gaussian_scale = torch.Tensor([temp_scale, ] * 7)

    def gaussian_step(self, reward=0.0):

        self.gaussian_optimizer.zero_grad()
        self.add_multi_gaussian_log_prob(self.actions)
        loss = -torch.sum(torch.stack(self.log_prob, dim=-1)) * reward
        loss.backward()


        for param in [self.gaussian_param_loc, ]:
            if param.requires_grad:
                link.all_reduce(param.grad.data)

        self.gaussian_optimizer.step()

        # broadcast gradients
        for param in [self.gaussian_param_loc]:
            link.broadcast(param, 0)

        # # reset
        del self.actions[:]
        del self.log_prob[:]


