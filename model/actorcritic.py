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
        :param ps: 原预测
        :param ps_: 新预测
        :param label: 源域标签
        :return:
        '''
        #源域损失
        origin = (ps > 0.5).float() * label
        after = (ps_ > 0.5).float() * label
        mask = (source_num == source).float()
        #惩罚把原来对的改错, 奖励把原来错的改对
        reward = torch.sum((after - origin),dim=-1) * mask

        #源域尽量疏松
        ps_ = F.softmax(ps_,dim=-1)
        reward = reward - torch.mean(torch.sum(ps_ * torch.log(ps_),dim=-1) * (source_num == source).float(),dim=-1)

        #目标域尽量紧凑
        pt_hot = F.one_hot(torch.argmax(ps_,dim=-1),11)
        reward = torch.sum(pt_hot * torch.log(ps_),dim=-1) * (source != source_num).float() + reward

        return reward #越大越好
