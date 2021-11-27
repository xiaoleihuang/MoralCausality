import torch
from loguru import logger
from experiment.metric import F1
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
import random
import ot
from model.classifier import Classifier
from model.actorcritic import Critic,CalReward

def train_b(encoder,classifier,traindata,testdata,optimizer1,optimizer2,args):

    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    best = 0
    best_epoch = 0
    for epoch in range(1, args.N_EPOCHS+1):

        encoder.train()
        classifier.train()
        for idx, batch in enumerate(traindata):

            x = batch.review_s.cuda()
            y = batch.label_s.float().cuda()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            embedding,_ = encoder(x)
            prediction = classifier(embedding)
            loss = F.binary_cross_entropy(prediction,y)
            loss.backward(retain_graph=True)
            optimizer1.step()
            optimizer2.step()

        precision, recall, f1 = eval(encoder,classifier,testdata,epoch)

        if best < f1:
            best = f1
            best_epoch = epoch
            a = precision, recall, f1

    logger.info('test epoch: {}, F1: {}, precision:{}. recall:{}'.format(best_epoch, a[-1], a[0], a[1]))


def train_dann(encoder,classifier,traindata,testdata,optimizer1,optimizer2,args):

    best = 0
    best_epoch = 0
    for epoch in range(1, args.N_EPOCHS+1):

        encoder.train()
        classifier.train()
        for idx, batch in enumerate(traindata):

            x_s = batch.review_s
            y_s = batch.label_s.float()
            x_t = batch.review_t
            y_t = batch.label_t.float()

            p = torch.tensor(epoch/(args.N_EPOCHS+1)).cuda()
            x_s = x_s.cuda()
            embedding,domain = encoder((x_s),p)
            x_s = x_s.cpu()
            prediction = classifier(embedding)
            y_s = y_s.cuda()
            loss = F.binary_cross_entropy(prediction,y_s)
            y_s = y_s.cpu()

            source = [0] * len(y_s) + [1] * len(y_t)
            source = torch.tensor(source)

            pad = torch.zeros(len(x_s),max(len(x_s),len(x_t)) - min(len(x_s),len(x_t)))
            if len(x_s)>len(x_t):
                x_t = torch.cat([x_s,pad],dim=-1)
            else:
                x_s = torch.cat([x_t,pad],dim=-1)
            x = torch.cat([x_s,x_t],dim=0)
            y = torch.cat([y_s,y_t],dim=0)
            data = torch.cat([x,y,source.unsqueeze(-1)],dim=-1).numpy()
            random.shuffle(data)
            data = torch.from_numpy(data).cuda()

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            embedding,domain = encoder(data[:,:-13],p)

            mask = (data[:,-1] == 0).float().cuda()
            loss1 = 0.1*F.binary_cross_entropy(domain[-1], mask)
            loss = loss1 + loss

            loss.backward()
            optimizer1.step()
            optimizer2.step()

        precision, recall, f1 = eval(encoder,classifier,testdata,epoch)

        if best < f1:
            best = f1
            best_epoch = epoch
            a = precision, recall, f1

    logger.info('test epoch: {}, F1: {}, precision:{}. recall:{}'.format(best_epoch, a[-1], a[0], a[1]))


def train_rl(encoder,classifier,traindata,testdata,optimizer1,optimizer2,args):

    optimizer = optim.Adam([encoder.parameters(),classifier.parameters()], args.lr)
    best = 0
    best_epoch = 0
    critic = Critic(args)
    critic = critic.cuda()
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=0.001, betas=(0.9, 0.99), eps=0.0000001)
    optimizer_rl = torch.optim.Adam(encoder.fc_domain.parameters(), lr=0.0001, betas=(0.9, 0.99), eps=0.0000001)

    for epoch in range(1, args.N_EPOCHS+1):

        encoder.train()
        classifier.train()
        for idx, batch in enumerate(traindata):

            x = batch.review.cuda()
            y = batch.label.float().cuda()
            optimizer.zero_grad()
            embedding,_ = encoder(x)
            prediction = classifier(embedding)
            loss = F.binary_cross_entropy(prediction,y)
            loss.backward()
            optimizer.step()

        for idx, batch in enumerate(traindata):

            x = batch.review.cuda()
            y = batch.label.cuda()
            source = batch.source.cuda()
            optimizer_rl.zero_grad()
            optimizer_critic.zero_grad()

            embedding,_ = encoder(x)
            prediction = classifier(embedding)
            action, logpro, entropy = \
                encoder.fc_domain.action, encoder.fc_domain.logpro, encoder.fc_domain.entrop
            value = CalReward().reward(prediction,y,source)
            prediction = critic(x,prediction)
            loss1 = F.mse_loss(prediction.squeeze(),value)
            loss1.backward(retain_graph=True)
            optimizer_critic.step()
            prediction = critic(x,prediction)
            loss2 = encoder.fc_domain.cal_loss(value,prediction,logpro,entropy)
            loss2.backward()
            optimizer_rl.step()

        precision, recall, f1 = eval(encoder,classifier,testdata,epoch)

        if best < f1:
            best = f1
            best_epoch = epoch
            a = precision, recall, f1

    logger.info('test epoch: {}, F1: {}, precision:{}. recall:{}'.format(best_epoch, a[-1], a[0], a[1]))

def train_mcd(extractor,classifier1,traindata,testdata,opt_e,opt_c1,args):

    '''
    Maximum Classifier Discrepancy for Unsupervised Domain Adaptation
    '''
    best = 0
    best_epoch = 0
    classifier2 = Classifier(classifier1.hidden_dim).cuda()
    opt_c2 = torch.optim.Adam(classifier2.parameters(), lr=args.lr)

    for epoch in range(1, args.N_EPOCHS+1):

        extractor.train()
        classifier1.train()
        classifier2.train()

        for idx, batch in enumerate(traindata):

            x = batch.review_s.cuda()
            y = batch.label_s.float().cuda()

            '''
            STEP A
            '''
            opt_e.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()

            src_feat, domain = extractor(x)
            preds_s1 = classifier1(src_feat)
            preds_s2 = classifier2(src_feat)

            loss_A =  F.binary_cross_entropy(preds_s1, y) + F.binary_cross_entropy(preds_s2, y)
            loss_A.backward()

            opt_e.step()
            opt_c1.step()
            opt_c2.step()

            '''
            STEP B
            '''
            x2 = batch.review_t.cuda()

            opt_e.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()

            src_feat,domain = extractor(x)
            preds_s1 = classifier1(src_feat)
            preds_s2 = classifier2(src_feat)
            loss_B =  F.binary_cross_entropy(preds_s1, y) + F.binary_cross_entropy(preds_s2, y)

            src_feat,domain = extractor(x2)
            preds_t1 = classifier1(src_feat)
            preds_t2 = classifier2(src_feat)

            loss_B  = loss_B - torch.mean(torch.abs(preds_t1 - preds_t2))
            loss_B.backward()

            opt_c1.step()
            opt_c2.step()

            opt_e.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()

            '''
            STEP C
            '''
            N = 4
            for i in range(N):
                feat_tgt,domain = extractor(x2)
                preds_t1 = classifier1(feat_tgt)
                preds_t2 = classifier1(feat_tgt)
                loss_C = torch.mean(torch.abs(preds_t1- preds_t2))
                loss_C.backward()
                opt_e.step()

                opt_e.zero_grad()
                opt_c1.zero_grad()
                opt_c2.zero_grad()

        precision, recall, f1 = eval(extractor,classifier1,testdata,epoch)
        if best < f1:
            best = f1
            best_epoch = epoch
            a = precision, recall, f1
        precision, recall, f1 = eval(extractor,classifier2,testdata,epoch)
        if best < f1:
            best = f1
            best_epoch = epoch
            a = precision, recall, f1

    logger.info('test epoch: {}, F1: {}, precision:{}. recall:{}'.format(best_epoch, a[-1], a[0], a[1]))


def train_jumbot(encoder,classifier,traindata,testdata,optimizer1,optimizer2,args):

    best = 0
    best_epoch = 0
    alpha = 0.01
    lambda_t = 0.5
    reg_m = 0.5
    #OH param : alpha = 0.01, lambda = 0.5, reg_m = 0.5
    #VisDA param : alpha = 0.005, lambda = 1., reg_m = 0.3
    for epoch in range(1, args.N_EPOCHS+1):

        encoder.train()
        classifier.train()
        for idx, batch in enumerate(traindata):

            x_s = batch.review_s.cuda()
            y_s = batch.label_s.float().cuda()
            x_t = batch.review_t.cuda()

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            g_xs, domain_s = encoder(x_s)
            pred_s = classifier(g_xs)

            g_xt, domain_t = encoder(x_t)
            pred_t = classifier(g_xt)

            loss_A =  F.binary_cross_entropy(pred_s, y_s)

            M_embed = torch.cdist(g_xs, g_xt)**2  # Term on embedded data
            M_sce = - torch.mm(y_s, torch.transpose(torch.log(pred_t), 0, 1))  # Term on labels
            M = alpha * M_embed + lambda_t * M_sce

            #OT computation
            a, b = ot.unif(g_xs.size()[0]), ot.unif(g_xt.size()[0])
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.detach().cpu().numpy(),0.01, reg_m=reg_m)
            pi = torch.from_numpy(pi).float().cuda()  # Transport plan between minibatches
            transfer_loss = torch.sum(pi * M)

            total_loss = loss_A + transfer_loss
            total_loss.backward()
            optimizer1.step()
            optimizer2.step()

        precision, recall, f1 = eval(encoder,classifier,testdata,epoch)
        if best < f1:
            best = f1
            best_epoch = epoch
            a = precision, recall, f1

    logger.info('test epoch: {}, F1: {}, precision:{}. recall:{}'.format(best_epoch, a[-1], a[0], a[1]))

class train_dta:

    @staticmethod
    def train(encoder,classifier,traindata,testdata,optimizer1,optimizer2,args):

        rampup_length = 30
        fc_delta = 0
        target_fc_consistency_weight = 0
        entmin_weight = 0
        source_cnn_consistency_weight = 0
        cls_balance_weight = 0

        for epoch in range(1, args.N_EPOCHS+1):

            encoder.train()
            classifier.train()
            for idx, batch in enumerate(traindata):

                x_s = batch.review_s.cuda()
                y_s = batch.label_s.float().cuda()
                x_t = batch.review_t.cuda()

                optimizer1.zero_grad()
                optimizer2.zero_grad()

                g_xt, domain_t = encoder(x_t)
                target_logits1 = classifier(g_xt[0])

                jacobian_for_cnn_adv_drop, jacobian_for_fc_adv_drop, clean_target_logits = train_dta().calculate_jacobians(
                    g_xt[1].detach(), target_logits1.detach(), classifier, encoder.module.drop_size,
                    )
                optimizer1.zero_grad()
                optimizer2.zero_grad()

                fc_drop_delta = min(1.0, (epoch + 1) / rampup_length) * fc_delta
                target_fc_dropout_mask, _ = train_dta().create_adversarial_dropout_mask(
                torch.ones_like(jacobian_for_fc_adv_drop),
                jacobian_for_fc_adv_drop, fc_drop_delta)

                _, target_predicted = clean_target_logits.max(1)

                target_logits_fc_drop = classifier(g_xt[2]*target_fc_dropout_mask)
                target_consistency_loss = target_fc_consistency_weight * F.kl_div(
                    target_logits_fc_drop,target_logits1)
                target_entropy_loss = entmin_weight * (-target_logits1 * torch.log(target_logits1))
                target_loss = target_consistency_loss + target_entropy_loss

                # Class balance
                cls_balance_loss = cls_balance_weight * -torch.mean(torch.log(torch.mean(target_logits1, 0) + 1e-6))
                target_loss = cls_balance_loss + target_loss

                target_loss.backward()

                # Source CE Loss
                source_features, _ = encoder(x_s)
                source_logits1, source_logits2 = classifier(source_features[0]), classifier(source_features[1])

                # Source pi model
                ce_loss = F.binary_cross_entropy(source_logits1, y_s)
                source_consistency_loss = 2 * source_cnn_consistency_weight * F.kl_div(source_logits2, source_logits1)
                source_loss = ce_loss + source_consistency_loss
                source_loss.backward()

                optimizer1.step()
                optimizer2.step()


    @staticmethod
    def calculate_jacobians(h, clean_logits, classifier, fc_mask_size):
        cnn_mask = torch.ones((*h.size()[:2], 1, 1)).to(h.device)
        fc_mask = torch.ones(cnn_mask.size(0), fc_mask_size).to(cnn_mask.device)
        cnn_mask.requires_grad = True
        fc_mask.requires_grad = True

        h_logits = classifier(cnn_mask * h, fc_mask)
        discrepancy = F.kl_div(h_logits, clean_logits)
        discrepancy.backward()

        return cnn_mask.grad.clone(), fc_mask.grad.clone(), h_logits

    @staticmethod
    def create_adversarial_dropout_mask(mask, jacobian, delta):
        """
        :param mask: shape [batch_size, ...]
        :param jacobian: shape [batch_size, ...]
        :param delta:
        :return:
        """
        num_of_units = int(torch.prod(torch.tensor(mask.size()[1:])).to(torch.float))
        change_limit = int(num_of_units * delta)
        mask = (mask > 0).to(torch.float)

        if change_limit == 0:
            return deepcopy(mask).detach(), torch.Tensor([]).type(torch.int64)

        # mask (mask=1 -> m = 1), (mask=0 -> m=-1)
        m = 2 * mask - torch.ones_like(mask)

        # sign of Jacobian  (J>0 -> s=1), (J<0 -> s=-1)
        s = torch.sign(jacobian)

        # remain (J>0, m=-1) and (J<0, m=1), which are candidates to be changed
        change_candidates = ((m * s) < 0).to(torch.float)

        # ordering abs_jacobian for candidates
        # the maximum number of the changes is "change_limit"
        # draw top_k elements ( if the top k element is 0, the number of the changes is less than "change_limit" )
        abs_jacobian = torch.abs(jacobian)
        candidate_abs_jacobian = (change_candidates * abs_jacobian).view(-1, num_of_units)
        topk_values, topk_indices = torch.topk(candidate_abs_jacobian, change_limit + 1)
        min_values = topk_values[:, -1]
        change_target_marker = (candidate_abs_jacobian > min_values.unsqueeze(-1)).view(mask.size()).to(torch.float)

        # changed mask with change_target_marker
        adv_mask = torch.abs(mask - change_target_marker)

        # normalization
        adv_mask = adv_mask.view(-1, num_of_units)
        num_of_undropped_units = torch.sum(adv_mask, dim=1).unsqueeze(-1)
        adv_mask = ((adv_mask / num_of_undropped_units) * num_of_units).view(mask.size())

        return adv_mask.clone().detach(), (adv_mask == 0).nonzero()[:, 1]


def eval(encoder,classifier,testdata,epoch):

    encoder.eval()
    classifier.eval()
    prediction_list = []
    target_list = []
    for idx, batch in enumerate(testdata):

        target_list += batch.label_s.tolist()
        x = batch.review_s.cuda()
        embedding,_ = encoder(x)
        prediction = classifier(embedding)
        prediction_list += prediction.cpu().tolist()

        target_list += batch.label_t.tolist()
        x = batch.review_t.cuda()
        embedding,_ = encoder(x)
        prediction = classifier(embedding)
        prediction_list += prediction.cpu().tolist()

    precision, recall, f1 = F1(prediction_list,target_list,epoch)

    return precision, recall, f1



