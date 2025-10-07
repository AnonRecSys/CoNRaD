import torch
import torch.nn as nn
from .BaseRecommenderColdDeb import BaseColdDebTrainer
from util.utils import next_batch_pairwise_cold_deb
import torch.nn.functional as F


class Heater(BaseColdDebTrainer):
    def __init__(
        self,
        args,
        training_data,
        standard_valid_data,
        deb_valid_data,
        all_valid_data,
        cold_valid_data,
        standard_test_data,
        deb_test_data,
        all_test_data,
        cold_test_data,
        user_num,
        item_num,
        user_idx,
        item_idx,
        device,
        user_content=None,
        item_content=None,
    ):
        super(Heater, self).__init__(
            args,
            training_data,
            standard_valid_data,
            deb_valid_data,
            all_valid_data,
            cold_valid_data,
            standard_test_data,
            deb_test_data,
            all_test_data,
            cold_test_data,
            user_num,
            item_num,
            user_idx,
            item_idx,
            device,
            user_content=user_content,
            item_content=item_content,
        )

        if args.save_output:
            self.repeat_id = args.repeater

        self.model = Heater_Learner(args, self.data, self.emb_size, device)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.args.reg
        )
        crit = torch.nn.MSELoss()
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(
                next_batch_pairwise_cold_deb(self.data, self.batch_size)
            ):
                user_idx, pos_idx, neg_idx = batch
                pos_pair_pred, diff_loss1 = model.heater_encoder_forward(
                    user_idx, pos_idx, True
                )
                neg_pair_pred, diff_loss2 = model.heater_encoder_forward(
                    user_idx, neg_idx, True
                )
                batch_pred = torch.cat((pos_pair_pred, neg_pair_pred), dim=0)
                batch_targets = torch.cat(
                    (torch.ones_like(pos_pair_pred), torch.zeros_like(neg_pair_pred)),
                    dim=0,
                )
                batch_loss = crit(batch_pred, batch_targets) + diff_loss1 + diff_loss2
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                """if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())"""

            with torch.no_grad():
                model.eval()
                now_user_emb, now_item_emb = self.model(False)
                self.user_emb = now_user_emb.clone()
                self.item_emb = now_item_emb.clone()
                if epoch % self.args.eval_freq == 0:
                    self.fast_evaluation_quiet(epoch, valid_type="cold")
            if epoch + 1 - self.bestPerformance[0] >= self.args.patience:
                break

        self.timer(start=False)
        model.eval()
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        if self.args.save_emb:
            torch.save(
                self.user_emb,
                f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_user_emb.pt",
            )
            torch.save(
                self.item_emb,
                f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_item_emb.pt",
            )

    def save(self):
        with torch.no_grad():
            now_best_user_emb, now_best_item_emb = self.model.forward(False)
            self.best_user_emb = now_best_user_emb.clone()
            self.best_item_emb = now_best_item_emb.clone()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


class Heater_Learner(nn.Module):
    def __init__(self, args, data, emb_size, device):
        super(Heater_Learner, self).__init__()
        self.args = args
        self.latent_size = emb_size
        self.device = device
        self.data = data

        if self.args.cold_object == "item":
            self.item_content = torch.cat(
                [F.normalize(torch.from_numpy(x).cuda()) for x in data.item_content],
                dim=1,
            )
        else:
            self.user_content = torch.tensor(
                self.data.mapped_user_content, dtype=torch.float32, requires_grad=False
            ).to(device)
        self.content_dim = self.item_content.shape[1]
        self.embedding_dict = self._init_model()
        self.heater_encoder = Heater_encoder(
            self.latent_size,
            0,
            self.content_dim,
            [200, 64],
            self.latent_size,
            self.args.alpha,
            self.args.n_expert,
            self.args.n_dropout,
        )
        self.embedding_dict = self._init_model()

    def _init_model(self):
        user_emb, item_emb = torch.load(
            f"../MMRec_embs/{self.args.dataset}/{self.args.backbone}.pt",
            map_location="cpu",
        )
        embedding_dict = nn.ParameterDict(
            {
                "user_emb": user_emb,
                "item_emb": item_emb,
            }
        )
        embedding_dict["user_emb"].requires_grad = False
        embedding_dict["item_emb"].requires_grad = False
        return embedding_dict

    def pair_score(self, uid, iid):
        user_emb = self.embedding_dict["user_emb"][uid]
        item_emb = self.embedding_dict["item_emb"][iid]
        return torch.sum(user_emb * item_emb, dim=1)

    def heater_encoder_forward(self, uid, iid, training=True):
        user_emb = self.embedding_dict["user_emb"][uid]
        item_emb = self.embedding_dict["item_emb"][iid]
        if self.args.cold_object == "item":
            item_content = self.item_content[iid]
            U_embedding, V_embedding, diff_loss, _ = self.heater_encoder.encode(
                user_emb, item_emb, None, item_content, training
            )
        else:
            user_content = self.user_content[uid]
            U_embedding, V_embedding, _, diff_loss = self.heater_encoder.encode(
                user_emb, item_emb, user_content, None
            )
        preds = U_embedding * V_embedding
        preds = torch.sum(preds, 1)
        return preds, diff_loss

    def forward(self, training=True):
        user_emb = self.embedding_dict["user_emb"]
        item_emb = self.embedding_dict["item_emb"]
        if self.args.cold_object == "item":
            item_content = self.item_content
            if training:
                item_content = item_content[self.data.warm_item_list]
            u_infer_emb, i_infer_emb, _, _ = self.heater_encoder.encode(
                user_emb, item_emb, None, item_content, training
            )
        else:
            user_content = self.user_content
            u_infer_emb, i_infer_emb, _, _ = self.heater_encoder.encode(
                user_emb, item_emb, user_content, None
            )
        return u_infer_emb, i_infer_emb


def l2_norm(para):
    return torch.sum(torch.pow(para, 2))


class DenseFC(nn.Module):
    def __init__(self, in_dim, model_select=[200, 100]):
        super(DenseFC, self).__init__()
        self.model_select = model_select
        self.linear1 = nn.Linear(in_dim, model_select[0])
        self.linear2 = nn.Linear(model_select[0], model_select[1])
        self.act = nn.Tanh()

    def forward(self, x):
        h1 = self.linear1(x)
        h1 = self.act(h1)
        h2 = self.linear2(h1)
        output = self.act(h2)
        return output


class Gate_func(nn.Module):
    def __init__(self, in_dim, n_expert):
        super(Gate_func, self).__init__()
        self.n_expert = n_expert
        self.linear = nn.Linear(in_dim, n_expert)

    def forward(self, x):
        gate = torch.tanh(self.linear(x))
        return gate


class Heater_encoder(nn.Module):
    def __init__(
        self,
        latent_rank_in,
        user_content_rank,
        item_content_rank,
        model_select,
        rank_out,
        alpha,
        n_expert,
        n_dropout,
    ):
        super(Heater_encoder, self).__init__()
        self.latent_rank_in = latent_rank_in
        self.user_content_rank = user_content_rank
        self.item_content_rank = item_content_rank
        self.model_select = model_select
        self.rank_out = rank_out
        self.alpha = alpha
        self.n_expert = n_expert
        self.n_dropout = n_dropout
        self.fc = DenseFC(self.item_content_rank, self.model_select)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.model_select[-1], self.model_select[-1]), nn.Tanh()
                )
                for _ in range(n_expert)
            ]
        )
        self.gate = Gate_func(self.item_content_rank, self.n_expert)
        self.out_linear_u = nn.Sequential(
            nn.Linear(self.model_select[-1], self.rank_out),
        )
        self.out_linear_i = nn.Sequential(
            nn.Linear(self.model_select[-1], self.rank_out),
        )

    def encode(self, Uin, Vin, Ucontent=None, Vcontent=None, training=True):
        u_last = Uin
        diff_user_loss = 0

        vcontent_gate = self.gate(Vcontent).unsqueeze(1)
        Vcontent_mapped = self.fc(Vcontent)
        vcontent_expert_list = []
        for expert in self.experts:
            tmp_expert = expert(Vcontent_mapped)
            vcontent_expert_list.append(tmp_expert.unsqueeze(1))
        vcontent_expert_concat = torch.cat(vcontent_expert_list, dim=1)
        Vcontent_last = torch.bmm(vcontent_gate, vcontent_expert_concat).squeeze(1)

        if training:
            Vin_filter = 1 - self.n_dropout
            diff_item_loss = self.alpha * torch.sum(
                torch.sum((Vcontent_last - Vin) ** 2, dim=1, keepdim=True)
            )
            v_last = Vin * Vin_filter + Vcontent_last * (1 - Vin_filter)
        else:
            Vin_filter, diff_item_loss = 0.0, 0.0
            v_last = Vcontent_last

        user_emb = self.out_linear_u(u_last)
        item_emb = self.out_linear_i(v_last)

        return user_emb, item_emb, diff_item_loss, diff_user_loss
