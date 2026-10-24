from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import random

from .mechanism_variants import parse_mechanism_variant, PLOTSlotAttention, PRIORPrototypeBank

class SAGAlignNet(nn.Module):
    def __init__(self, args, num_classes=7567):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.num_prototypes = args.K
        if self.num_prototypes <= 0:
            raise ValueError("K must be positive")
        if args.m <= 0:
            raise ValueError("m must be positive")
        if args.neighbor_weight < 0.0:
            raise ValueError("neighbor_weight must be non-negative")
        if not 0.0 <= args.lambda_ <= 1.0:
            raise ValueError("lambda must be in [0, 1]")
        self._set_task()
        self.cprod_variant, self.masalign_variant = parse_mechanism_variant(getattr(args, "mechanism_variant", "native"))
        if self.cprod_variant != "native" and "sdm" not in self.current_task:
            raise ValueError("CProD prototype variants require sdm in loss_names")
        if self.masalign_variant != "native" and "mlm" not in self.current_task:
            raise ValueError("MasAlign variants require mlm in loss_names")
        print("Mechanism variants: CProD={}, MasAlign={}".format(self.cprod_variant, self.masalign_variant))

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        self.discriminator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
            )

        if "matching_label" in self.current_task:
            '''
            self.classifier = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, self.num_classes))]))
            nn.init.normal_(self.classifier.dense.weight, std=0.001)
            nn.init.constant_(self.classifier.dense.bias, val=0.0)
            nn.init.normal_(self.classifier.fc.weight, std=0.001)
            nn.init.constant_(self.classifier.fc.bias, val=0.0)
            '''
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)
        if 'mlm' or 'sdm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

            if self.cprod_variant == "native":
                self.query_embed_image = nn.Parameter(
                    torch.randn(1, self.num_prototypes, self.embed_dim)
                )
                self.cross_attn_pgu = nn.MultiheadAttention(
                    self.embed_dim, self.embed_dim // 64, batch_first=True
                )
                nn.init.normal_(self.cross_attn_pgu.in_proj_weight, std=attn_std)
                nn.init.normal_(self.cross_attn_pgu.out_proj.weight, std=proj_std)
                self.ln_t_pgu = LayerNorm(self.embed_dim)
                self.ln_i_pgu = LayerNorm(self.embed_dim)
                self.ln_p_pgu = LayerNorm(self.embed_dim)
                self.fc_layers = nn.ModuleList([
                    nn.Linear(self.embed_dim, self.embed_dim)
                    for _ in range(self.num_prototypes)
                ])
            elif self.cprod_variant == "plot":
                self.prototype_slots = nn.Parameter(
                    torch.randn(1, self.num_prototypes, self.embed_dim)
                    / self.embed_dim ** 0.5
                )
                self.plot_prototype_attention_i = PLOTSlotAttention(self.embed_dim)
                self.plot_prototype_attention_t = PLOTSlotAttention(self.embed_dim)
            elif self.cprod_variant == "prior":
                self.prior_prototype_bank = PRIORPrototypeBank(
                    self.embed_dim, self.num_prototypes, temperature=0.9
                )

            self.mlm_head_6 = nn.Sequential(
                OrderedDict([
                    ('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                    ('gelu', QuickGELU()),
                    ('ln', LayerNorm(self.embed_dim)),
                    ('fc', nn.Linear(self.embed_dim, self.num_prototypes)),
                ])
            )

            if self.masalign_variant == "native":
                self.cross_attn_error = nn.MultiheadAttention(
                    self.embed_dim, self.embed_dim // 64, batch_first=True
                )
                self.ln_pre_t_e = LayerNorm(self.embed_dim)
                self.ln_pre_i_e = LayerNorm(self.embed_dim)
                nn.init.normal_(self.cross_attn_error.in_proj_weight, std=attn_std)
                nn.init.normal_(self.cross_attn_error.out_proj.weight, std=proj_std)
                self.mlm_head_e = nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.ReLU(),
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.ReLU(),
                    nn.Linear(self.embed_dim, 1),
                )
            elif self.masalign_variant == "rasa":
                self.rasa_itm_head = nn.Linear(self.embed_dim, 2)
                self.rasa_prd_head = nn.Linear(self.embed_dim, 2)
                self.rasa_mrtd_head = nn.Linear(self.embed_dim, 2)


    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')


    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def cross_former_error(self, q, k, v):
        x = self.cross_attn_error(
                self.ln_pre_t_e(q),
                self.ln_pre_i_e(k),
                self.ln_pre_i_e(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()


    def rasa_relation_losses(self, image_feats, text_feats, i_feats, t_feats, pids):
        same = pids[:, None].eq(pids[None, :])
        batch_size = pids.size(0)
        valid = (~same).any(dim=1)
        zero = (self.rasa_itm_head.weight.sum()
                + self.rasa_prd_head.weight.sum()) * 0.0
        positive = self.cross_former(
            text_feats, image_feats, image_feats
        )[:, 0, :]
        pitm_loss = zero

        if valid.any():
            rows = torch.nonzero(valid, as_tuple=False).flatten()
            similarities = (
                F.normalize(i_feats.float(), dim=1)
                @ F.normalize(t_feats.float(), dim=1).t()
            )
            with torch.no_grad():
                text_weights = F.softmax(similarities, dim=1).masked_fill(same, 0.0)
                text_weights = text_weights / text_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
                image_weights = F.softmax(similarities, dim=0).t().masked_fill(same.t(), 0.0)
                image_weights = image_weights / image_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
                neg_text_idx = torch.multinomial(text_weights[rows], 1).flatten()
                neg_image_idx = torch.multinomial(image_weights[rows], 1).flatten()

            negative_text = self.cross_former(
                text_feats[neg_text_idx], image_feats[rows], image_feats[rows]
            )[:, 0, :]
            negative_image = self.cross_former(
                text_feats[rows], image_feats[neg_image_idx], image_feats[neg_image_idx]
            )[:, 0, :]
            itm_logits = torch.cat((
                self.rasa_itm_head(positive[rows]),
                self.rasa_itm_head(negative_text),
                self.rasa_itm_head(negative_image),
            ))
            itm_labels = torch.cat((
                torch.ones(rows.numel(), dtype=torch.long, device=pids.device),
                torch.zeros(2 * rows.numel(), dtype=torch.long, device=pids.device),
            ))
            pitm_loss = F.cross_entropy(itm_logits.float(), itm_labels)

        eye = torch.eye(batch_size, dtype=torch.bool, device=pids.device)
        alt_mask = same & ~eye
        alt_rows = torch.nonzero(alt_mask.any(dim=1), as_tuple=False).flatten()
        prd_loss = zero
        if alt_rows.numel():
            alt_idx = alt_mask[alt_rows].float().argmax(dim=1)
            alt_positive = self.cross_former(
                text_feats[alt_idx], image_feats[alt_rows], image_feats[alt_rows]
            )[:, 0, :]
            prd_logits = torch.cat((
                self.rasa_prd_head(positive[alt_rows]),
                self.rasa_prd_head(alt_positive),
            ))
            prd_labels = torch.cat((
                torch.zeros(alt_rows.numel(), dtype=torch.long, device=pids.device),
                torch.ones(alt_rows.numel(), dtype=torch.long, device=pids.device),
            ))
            prd_loss = F.cross_entropy(prd_logits.float(), prd_labels)
        return pitm_loss, prd_loss

    def rasa_mrtd_loss(self, logits, batch, image_feats):
        batch_size, text_length, vocab_size = logits.shape
        topk = min(32, vocab_size)
        candidates = torch.topk(logits.float(), topk, dim=-1).indices
        sampled = torch.randint(
            topk, (batch_size, text_length, 1), device=logits.device
        )
        replacements = torch.gather(candidates, -1, sampled).squeeze(-1)
        corrupted = batch['mlm_ids'].clone()
        masked = batch['mlm_labels'].ne(0)
        corrupted[masked] = replacements[masked]
        labels = corrupted.ne(batch['caption_ids']).long()

        tokens = self.base_model.encode_text(corrupted)
        fused = self.cross_former(tokens, image_feats, image_feats)
        pred = self.rasa_mrtd_head(fused).float()
        positions = torch.arange(text_length, device=logits.device).unsqueeze(0)
        eos = batch['caption_ids'].argmax(dim=-1).unsqueeze(1)
        valid = batch['caption_ids'].ne(0) & positions.gt(0) & positions.lt(eos)
        if valid.any():
            return F.cross_entropy(pred[valid], labels[valid])
        return self.rasa_mrtd_head.weight.sum() * 0.0

    def _prototype_list(self, prototypes):
        return [prototypes[:, idx, :] for idx in range(self.num_prototypes)]

    def pgu_encode_i(self, i_feature):
        if self.cprod_variant == "native":
            query_embed = self.query_embed_image.repeat(
                i_feature.size(0), 1, 1
            ).half()
            i_pgu = self.cross_attn_pgu(
                self.ln_p_pgu(query_embed),
                self.ln_i_pgu(i_feature),
                self.ln_i_pgu(i_feature),
                need_weights=False,
            )[0]
            outputs = [
                fc_layer(i_pgu[:, idx, :])
                for idx, fc_layer in enumerate(self.fc_layers)
            ]
            return outputs, i_feature.float().sum() * 0.0

        visual_tokens = i_feature[:, 1:, :]
        if self.cprod_variant == "plot":
            prototypes = self.plot_prototype_attention_i(
                visual_tokens, self.prototype_slots
            )
            return self._prototype_list(prototypes), i_feature.float().sum() * 0.0

        valid = torch.ones(
            visual_tokens.shape[:2], dtype=torch.bool, device=visual_tokens.device
        )
        prototypes, reconstruction_loss = self.prior_prototype_bank(
            visual_tokens, valid
        )
        return self._prototype_list(prototypes), reconstruction_loss

    def pgu_encode_t(self, t_feature, caption_ids):
        if self.cprod_variant == "native":
            query_embed = self.query_embed_image.repeat(
                t_feature.size(0), 1, 1
            ).half()
            t_pgu = self.cross_attn_pgu(
                self.ln_p_pgu(query_embed),
                self.ln_t_pgu(t_feature),
                self.ln_t_pgu(t_feature),
                need_weights=False,
            )[0]
            outputs = [
                fc_layer(t_pgu[:, idx, :])
                for idx, fc_layer in enumerate(self.fc_layers)
            ]
            return outputs, t_feature.float().sum() * 0.0

        text_tokens = t_feature[:, 1:, :]
        token_ids = caption_ids[:, 1:]
        positions = torch.arange(
            1, caption_ids.size(1), device=caption_ids.device
        ).unsqueeze(0)
        eos_positions = caption_ids.argmax(dim=-1).unsqueeze(1)
        valid = token_ids.ne(0) & positions.lt(eos_positions)

        if self.cprod_variant == "plot":
            prototypes = self.plot_prototype_attention_t(
                text_tokens, self.prototype_slots, padding_mask=~valid
            )
            return self._prototype_list(prototypes), t_feature.float().sum() * 0.0

        prototypes, reconstruction_loss = self.prior_prototype_bank(
            text_tokens, valid
        )
        return self._prototype_list(prototypes), reconstruction_loss

    def forward(self, batch, epoch, store1, store2, label_store):
        ret = dict()
        images = batch['images']
        caption_ids = batch['caption_ids'] ###############原始##############
        #caption_ids = batch['mlm_ids']
        #########error modeling#########
        t_index = caption_ids.argmax(dim=-1)

        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        BCE_loss = nn.BCEWithLogitsLoss()
        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})
        # output_i = [i_feats.unsqueeze(1)] * 6
        # output_t = [t_feats.unsqueeze(1)] * 6
        if 'adv' in self.current_task:
            gi_feats = F.normalize(i_feats, dim=1)
            gt_feats = F.normalize(t_feats, dim=1)
            i_dis = self.discriminator((gi_feats).half()).float()
            t_dis = self.discriminator((gt_feats).half()).float()
            y_real = torch.ones(i_feats.size(0)).to(i_feats.device)
            y_fake = torch.zeros(i_feats.size(0)).to(i_feats.device)
            BCE_loss = nn.BCEWithLogitsLoss()
            D_real_loss = BCE_loss(i_dis.squeeze(), y_real)
            D_fake_loss = BCE_loss(t_dis.squeeze(), y_fake)
            D_train_loss = D_real_loss + D_fake_loss
            ret.update({'adv_loss': D_train_loss})
        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_js_sdm(
                i_feats, t_feats, batch['pids'], logit_scale,
                gamma=self.args.js_gamma,
            )})
            ##############pgu##############
            output_i, proto_recon_i = self.pgu_encode_i(image_feats)
            output_t, proto_recon_t = self.pgu_encode_t(
                text_feats, caption_ids
            )
            if self.cprod_variant == "prior":
                ret.update({
                    'prior_proto_loss': 10.0 * (proto_recon_i + proto_recon_t)
                })
            output_i_c = torch.cat(output_i,dim=1).float()
            output_t_c = torch.cat(output_t,dim=1).float()
            local_prototype_loss = objectives.compute_js_sdm(
                output_i_c, output_t_c, batch['pids'], logit_scale,
                gamma=self.args.js_gamma,
            )
            ret.update({
                'xld_loss': local_prototype_loss
            })
            total_6 = 0
            for i in range(self.num_prototypes):
                image_6 = self.mlm_head_6(output_i[i]).float()
                text_6 = self.mlm_head_6(output_t[i]).float()
                lable_6 = (torch.ones(i_feats.size(0),dtype=torch.int) * i).long().to(i_feats.device)
                total_6 += (objectives.compute_mlm_6(image_6,lable_6) + objectives.compute_mlm_6(text_6,lable_6))
            ret.update({'hc_loss':total_6})
        if self.masalign_variant == "rasa":
            pitm_loss, prd_loss = self.rasa_relation_losses(
                image_feats, text_feats, i_feats, t_feats, batch['pids']
            )
            ret.update({'rasa_pitm_loss': pitm_loss, 'rasa_prd_loss': prd_loss})
        if 'matching_label' in self.current_task:
            numm = i_feats.size(0)
            zero_tensor_i = torch.zeros(numm, self.num_classes).to(i_feats.device)
            zero_tensor_t = torch.zeros(numm, self.num_classes).to(i_feats.device)
            for i, index in enumerate(batch['pids']):
                        zero_tensor_i[i, index] = self.args.alpha
                        zero_tensor_t[i, index] = self.args.alpha

            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()

            if epoch == 1:
                ret.update({
                    'matching_label_loss': objectives.compute_matching_label(
                        image_logits, text_logits, batch['pids']
                    )
                })

            else:
                store_i = torch.cat(store1,dim=0)   ########(68000,6,512)
                store_t = torch.cat(store2,dim=0)
                stored_labels = torch.cat(label_store, dim=0)
                for i in range(self.num_prototypes):
                    sim1 = F.normalize(output_i[i], p=2, dim=1) @ F.normalize(store_i[:,i,:], p=2, dim=1).t()  ######(64,68000)
                    sim2 = F.normalize(output_t[i], p=2, dim=1) @ F.normalize(store_t[:,i,:], p=2, dim=1).t()

                    effective_m = min(self.args.m, store_i.size(0))
                    topk_values_i, topk_indice_i = torch.topk(sim1, k=effective_m, dim=1)
                    topk_values_t, topk_indice_t = torch.topk(sim2, k=effective_m, dim=1)

                    shot_i = torch.gather(stored_labels.unsqueeze(0).repeat(numm, 1), dim=1, index=topk_indice_i)
                    shot_t = torch.gather(stored_labels.unsqueeze(0).repeat(numm, 1), dim=1, index=topk_indice_t)

                    neighbor_src_i = torch.full_like(
                        shot_i, self.args.neighbor_weight, dtype=zero_tensor_i.dtype
                    )
                    neighbor_src_t = torch.full_like(
                        shot_t, self.args.neighbor_weight, dtype=zero_tensor_t.dtype
                    )
                    soft_label_i = zero_tensor_i.scatter_add(
                        1, shot_i, neighbor_src_i
                    )
                    soft_label_t = zero_tensor_t.scatter_add(
                        1, shot_t, neighbor_src_t
                    )

                lambda_value = self.args.lambda_
                hard = objectives.compute_matching_label(
                    image_logits, text_logits, batch['pids']
                )
                soft = objectives.compute_soft_matching_label(
                    image_logits, text_logits, soft_label_i, soft_label_t
                )
                ret.update({
                    'matching_label_loss':
                        (1.0 - lambda_value) * hard + lambda_value * soft
                })


            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']      ########原始########
            mlm_feats = self.base_model.encode_text(mlm_ids)
            #o_text = self.base_model.encode_text(batch['caption_ids'])   #####add
            #mlm_feats = text_feats                                       #####add
            x = self.cross_former(mlm_feats, image_feats, image_feats)

            if self.masalign_variant == "native":
                mlme_pre = self.mlm_head_e(x).reshape(-1)

            x = self.mlm_head(x)
            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)

            if self.masalign_variant == "native":
                error_labels0 = mlm_labels.ne(0).long()
                loss_pre0 = BCE_loss(mlme_pre.float(), error_labels0.float())
                ret.update({'mlm2_loss': loss_pre0 * self.args.mlm_loss_weight})

            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            scores = x.float().reshape(-1, self.args.vocab_size)

            pred = scores.max(1)[1] #一维tesor 616
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

            if self.masalign_variant == "native":
                error_list = []
                for i in range(batch['mlm_ids'].size(0)):
                    mlm_i = batch['mlm_ids'][i].clone()
                    mlm_lain = torch.nonzero(batch['mlm_labels'][i]).squeeze()
                    top_values, top_indices = torch.topk(x[i].float(), 32, dim=1, largest=True, sorted=True)
                    random_indices = torch.randint(low=0, high=32, size=(77,))
                    # 对 tensor 进行切片
                    result = top_indices[torch.arange(77), random_indices]
                    mlm_i[mlm_lain] = result[mlm_lain]
                    if mlm_lain.dim() == 0:
                        mlm_lain = mlm_lain.unsqueeze(0)
                    for index in mlm_lain:
                        if mlm_i[index].item() == batch['caption_ids'][i][index].item():
                            if top_indices[index][0].item() == batch['caption_ids'][i][index].item():
                                mlm_i[index] = top_indices[index][1]
                            else:
                                mlm_i[index] = top_indices[index][0]
                    error_list.append(mlm_i)

                error_text = torch.stack(error_list)

                error_feats = self.base_model.encode_text(error_text)
                error_x = self.cross_former_error(error_feats, image_feats, image_feats)
                '''
                error_pre = (self.mlm_head_e(error_x)).reshape(-1, 2)

                error_x = self.mlm_head(error_x)
                scores_e = error_x.float().reshape(-1, self.args.vocab_size)

                error_labels1 = torch.where(mlm_labels != 0, 1, mlm_labels)
                error_labels2 = error_labels1 ^ 1
                error_labels = torch.cat((error_labels1.unsqueeze(1), error_labels2.unsqueeze(1)), dim=1)

                loss_pre = F.cross_entropy(error_pre, error_labels.float())/0.1
                '''
                error_pre = (self.mlm_head_e(error_x)).reshape(-1)

                error_x = self.mlm_head(error_x)
                scores_e = error_x.float().reshape(-1, self.args.vocab_size)

                error_labels1 = torch.where(mlm_labels != 0, 1, mlm_labels)
                loss_pre = BCE_loss(error_pre.float(), error_labels1.float())

                ret.update({'mlmpre_loss': loss_pre*self.args.mlm_loss_weight})
                ret.update({'mlmerror_loss': objectives.compute_mlm(scores_e, mlm_labels)*self.args.mlm_loss_weight})
            elif self.masalign_variant == "rasa":
                mrtd = self.rasa_mrtd_loss(x.float(), batch, image_feats)
                ret.update({'rasa_mrtd_loss': mrtd * self.args.mlm_loss_weight})
            ###########################
            '''
            ###########global cat############
            add_feats = mlm_feats + image_feats[:, 0, :].unsqueeze(1)
            add_feats = self.cross_gcat_linear(add_feats)
            add_feats = add_feats.permute(1, 0, 2)
            add_feats = self.cross_modal_transformer(add_feats)
            add_feats = add_feats.permute(1, 0, 2)
            add_feats = self.ln_post(add_feats)
            add_feats = self.mlm_head(add_feats)
            scores = add_feats.float().reshape(-1, self.args.vocab_size)
            #mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm2_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})
            ##################################
            '''
            '''
            remain_image_feats = image_feats[:, 1:, :]
            #query_embed = self.query_embed_image.repeat(image_feats.size(0), 1, 1).half()

            output_i_end = self.pgu_encode_i(remain_image_feats,i_feats)
            output_t_end = self.pgu_encode_t(mlm_feats,t_feats)

            #output_i_end = self.ln_end_pgu(reshaped_i).float()
            #output_t_end = self.ln_end_pgu(reshaped_t).float()
            ret.update({'xld_loss':objectives.compute_sdm(output_i_end, output_t_end, batch['pids'], logit_scale)})
            '''
            ##############pgu##############


        return ret, torch.stack(output_i, dim=1), torch.stack(output_t, dim=1)


def build_model(args, num_classes=7567):
    model = SAGAlignNet(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
