import torch.nn as nn
import torch
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self,args,hidden_dim=256):
        super(Critic, self).__init__()

        self.fc = nn.Linear(hidden_dim*2, args.OUTPUT_DIM)
        self.pre = nn.Linear(3*args.OUTPUT_DIM, 1)

    def forward(self,x,p,p_):

        x = self.fc(x)
        reward = self.pre(torch.cat((x,p,p_),dim=-1))

        return reward


class CalReward():
    @staticmethod
    def reward(ps,ps_, label, source_num,source): #在不同epoch中训练结果的稳定性

        '''

        :param ps: 原始源域预测
        :param ps_: 新的源域预测
        :param label: 源域标签
        :param pt: rl目标域预测
        :return:
        '''

        weight = (torch.argmax(ps,dim=-1) == torch.argmax(label,dim=-1)).float()
        pss = F.softmax(ps, dim=-1)
        reward = torch.sum(-pss * torch.log(pss),dim=-1)
        reward = reward - torch.sum(ps_ * torch.log(ps),dim=-1)
        reward = reward * weight * (source_num == source).float()

        pt = F.softmax(ps_, dim=-1)
        pt_hot = F.one_hot(torch.argmax(pt,dim=-1),11)
        reward = torch.sum(pt_hot * torch.log(pt),dim=-1) * (source != source_num).float() + reward

        return reward #越大越好
