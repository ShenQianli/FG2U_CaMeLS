# %%
from copy import deepcopy
import torch.nn as nn
import torch
from torch.func import jvp, vmap, functional_call, grad, vjp
from transformers import GPT2Model, GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer, AutoModelForCausalLM
import os
from util import weighted_lm_loss, weighted_lm_loss_with_outputs, create_colored_text, get_pos_from_toks, \
    get_nes_from_toks
import numpy as np
from tqdm import tqdm
import higher
from collections import defaultdict
from util import kl_loc_loss, CACHE_DIR
import matplotlib.pyplot as plt
import torch.nn.functional as f
from omegaconf import OmegaConf
import pandas as pd
import spacy
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import pdb

default_config = OmegaConf.create({
    'log_dir': './logs',
    'inner_lr': 5e-3,
    'num_inner_steps': 1,
    'c_kl': .1,
    'norm': 2,
    'normalize': False,
    'non_linearity': 'softplus',
    'pretrained_model': 'distilgpt2',
    'c_norm': 0.1,
    'freeze_base': False,
    'normalize': False,
    'log_stepwise_metrics': False,
    'norm_from_one': True
})

# "----------------------------------------------------------------": 10097
IGNORED_INDEX = 10097
MAX_BATCH_SIZE = 6
TRUNCATED_STEPS = 6


# super class for any model which takes text outputs weights for each token
class WeightingModel(nn.Module):
    def __init__(self, config=default_config,
                 device_='cuda'):
        # if base_lm, we do not load a base_lm
        super().__init__()
        self.config = config
        self._log_dir = config.log_dir if 'log_dir' in config else default_config.log_dir
        self.base_lm = None
        self.device = device_
        self.inner_lr = config.inner_lr if 'inner_lr' in config else default_config.inner_lr
        self.num_inner_steps = config.num_inner_steps if 'num_inner_steps' in config else default_config.num_inner_steps
        self.c_kl = config.c_kl if 'c_kl' in config else default_config.c_kl
        self.c_norm = config.c_norm if 'c_norm' in config else default_config.c_norm
        self.norm = config.norm if 'norm' in config else default_config.norm
        self.log_stepwise_metrics = config.log_stepwise_metrics if 'log_stepwise_metrics' in config else default_config.log_stepwise_metrics
        self.norm_from_one = config.norm_from_one if 'norm_from_one' in config else default_config.norm_from_one

    def get_optimizer(self, outer_lr):
        raise NotImplementedError

    def set_base_lm(self, base_lm):
        self.base_lm = base_lm.to(self.device)

    def set_inner_lr(self, inner_lr):
        self.inner_lr = inner_lr

    def forward(self, x, attention_mask=None, index=None):
        raise NotImplementedError

    def get_updated_model(self, batch, base_lm, higher_grad, sequential_update=False, trgu=False):
        # modifying the base_lm via fine tuning on batch using models weight_function
        # the thing we return is a fmodule
        # copy_initial_weights whether the gradient from returned
        # sequential_update: whether to update the model sequentially or all at once
        # retuens the updated model and model weights on the batch
        targets = batch['text_ids'].clone()
        targets[batch['text_attention'] != 1] = -100
        optimizer = torch.optim.SGD(list(base_lm.parameters()), lr=self.inner_lr)
        if trgu:
            L = len(batch['text_ids']) - TRUNCATED_STEPS
            with torch.no_grad():
                weights = self(batch['text_ids'][:L], batch['text_attention'][:L], idx=batch['idx'][:L])
            for i in range(len(weights)):
                loss = weighted_lm_loss(base_lm, batch['text_ids'][i:i + 1], targets[i:i + 1],
                                        batch['text_attention'][i:i + 1], weights[i:i + 1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            weights = self(batch['text_ids'][L:], batch['text_attention'][L:], idx=batch['idx'][L:])
        else:
            if higher_grad:
                weights = self(batch['text_ids'], batch['text_attention'], idx=batch['idx'])
            else:
                with torch.no_grad():
                    weights = self(batch['text_ids'], batch['text_attention'], idx=batch['idx'])
        with higher.innerloop_ctx(base_lm, optimizer, copy_initial_weights=True, track_higher_grads=higher_grad) as (
                f_base_lm, diffopt):
            # copy_initial_weights=True means the gradient will not flow back to initial weights.
            # if two steps, add extra looping
            for _ in range(self.num_inner_steps):
                if sequential_update:
                    if trgu:
                        L = len(batch['text_ids']) - TRUNCATED_STEPS
                        for i in range(L, len(batch['text_ids'])):
                            loss = weighted_lm_loss(f_base_lm, batch['text_ids'][i:i + 1], targets[i:i + 1],
                                                    batch['text_attention'][i:i + 1], weights[i - L:i - L + 1])
                            diffopt.step(loss)
                    else:
                        for i in range(len(batch['text_ids'])):
                            loss = weighted_lm_loss(f_base_lm, batch['text_ids'][i:i + 1], targets[i:i + 1],
                                                    batch['text_attention'][i:i + 1], weights[i:i + 1])
                            diffopt.step(loss)
                else:
                    loss = weighted_lm_loss(f_base_lm, batch['text_ids'], targets, batch['text_attention'], weights)
                    diffopt.step(loss)
            return f_base_lm, weights

    @torch.no_grad()
    def forward_mode_step(self, update_batch, loc_batches={}, base_lm=None, train=True, sequential_update=True,
                          num_vs=2, update=False, grad_acc_steps=1):
        assert sequential_update
        if base_lm is None:
            base_lm = self.base_lm

        text_labels = update_batch['text_ids'].clone()
        # -100 will trigger vmap error
        text_labels[update_batch['text_attention'] != 1] = IGNORED_INDEX
        qa_labels = update_batch['qa_target_ids'].clone()
        qa_labels[qa_labels == -100] = IGNORED_INDEX

        # init_text_loss = base_lm(input_ids=update_batch['text_ids'], attention_mask=update_batch['text_attention'],
        #                          labels=text_labels).loss
        init_qa_outputs = base_lm(input_ids=update_batch['qa_ids'], attention_mask=update_batch['qa_attention'],
                                  labels=update_batch['qa_target_ids'])

        def _inner_loss_func(_inner_params, _text_ids, _targets, _text_attention, _weights):
            outputs = functional_call(
                base_lm, _inner_params, args=(),
                kwargs=dict(input_ids=_text_ids, attention_mask=_text_attention, labels=_targets)
            )
            loss = weighted_lm_loss_with_outputs(outputs, _text_ids, _targets, _weights, ignore_index=IGNORED_INDEX)
            return loss, outputs

        # important to use torch.no_grad() here
        @torch.no_grad()
        def _forward(_meta_params):
            inner_params = dict(base_lm.named_parameters())
            base_lm.train()
            batch_size = len(update_batch['text_ids'])
            if batch_size < MAX_BATCH_SIZE:
                weights = functional_call(self, _meta_params,
                                          (update_batch['text_ids'], update_batch['text_attention'],
                                           update_batch['idx']))
            else:
                leftover = batch_size % MAX_BATCH_SIZE
                weights = []
                for i in range(0, batch_size, MAX_BATCH_SIZE):
                    shift = leftover if i + MAX_BATCH_SIZE > batch_size else MAX_BATCH_SIZE
                    weights.append(
                        functional_call(self, _meta_params,
                                        (update_batch['text_ids'][i:i + shift],
                                         update_batch['text_attention'][i:i + shift],
                                         update_batch['idx'][i:i + shift]))
                    )
                    torch.cuda.empty_cache()
                weights = torch.cat(weights, dim=0)
            for _ in range(self.num_inner_steps):
                for i in range(len(update_batch['text_ids'])):
                    grads = grad(
                        lambda _inner_params: _inner_loss_func(
                            _inner_params, update_batch['text_ids'][i:i + 1], text_labels[i:i + 1],
                            update_batch['text_attention'][i:i + 1], weights[i:i + 1]
                        )[0]
                    )(inner_params)
                    # perform gradient descent
                    for k, v in inner_params.items():
                        inner_params[k] = inner_params[k] - self.inner_lr * grads[k]
                    del grads
                    torch.cuda.empty_cache()
            base_lm.eval()
            qa_outputs = functional_call(
                base_lm, inner_params, args=(),
                kwargs=dict(input_ids=update_batch['qa_ids'], attention_mask=update_batch['qa_attention'],
                            labels=qa_labels)
            )
            qa_loss = weighted_lm_loss_with_outputs(
                qa_outputs, update_batch['qa_ids'], qa_labels, weights=None, ignore_index=IGNORED_INDEX
            )

            metrics = {
                'qa_loss': qa_loss.detach(),
                'qa_gain': init_qa_outputs.loss.detach() - qa_loss.detach()
            }

            total_loss = qa_loss
            torch.cuda.empty_cache()

            if self.c_kl != 0:
                for name, loc_batch in loc_batches.items():
                    loc_text_labels = loc_batch['loc_ids'].clone()
                    loc_text_labels[loc_batch['loc_attention'] != 1] = IGNORED_INDEX
                    init_loc_qa_outputs = base_lm(input_ids=loc_batch['loc_ids'],
                                                  attention_mask=loc_batch['loc_attention'],
                                                  labels=loc_text_labels)
                    post_loc_qa_outputs = functional_call(
                        base_lm, inner_params, args=(),
                        kwargs=dict(input_ids=loc_batch['loc_ids'], attention_mask=loc_batch['loc_attention'],
                                    labels=loc_text_labels)
                    )
                    kl_div = kl_loc_loss(init_loc_qa_outputs.logits, post_loc_qa_outputs.logits, loc_batch['loc_mask'])
                    metrics[f'{name}_kl_div'] = kl_div.detach()
                    metrics[f'{name}_loc_gain'] = init_loc_qa_outputs.loss.detach() - post_loc_qa_outputs.loss.detach()
                    total_loss += self.c_kl * kl_div / len(loc_batches)
                    del init_loc_qa_outputs, post_loc_qa_outputs
                    torch.cuda.empty_cache()

            if (self.norm is not None):
                weight_norm = (
                        ((weights ** self.norm) * f.normalize(update_batch['text_attention'] * 1., p=1, dim=1)).sum(
                            1) ** (1 / self.norm))
                avg_norm = weight_norm.mean()
                weight_norm_from_one = ((((weights - 1) ** self.norm) * f.normalize(update_batch['text_attention'] * 1.,
                                                                                    p=1, dim=1)).sum(1) ** (
                                                1 / self.norm))
                avg_norm_from_one = weight_norm_from_one.mean()
                # weight_norm = torch.norm(weights, p = self.norm, dim = 1).mean()
                metrics[f'L{self.norm}-norm'] = avg_norm.detach()
                metrics[f'L{self.norm}-norm_from_one'] = avg_norm_from_one.detach()
                if self.c_norm != 0:
                    total_loss += self.c_norm * (avg_norm_from_one if self.norm_from_one else avg_norm)

            torch.cuda.empty_cache()

            metrics['total_loss'] = total_loss.detach()

            return total_loss, (metrics, inner_params)

        meta_params = dict((k, v) for k, v in self.named_parameters())
        vs = rademacher_like(meta_params, num_vs, device=self.device)
        in_dims = dict((k, 0) for k, v in vs.items())
        # pdb.set_trace()
        total_loss, jvp_output, (metrics, updated_params) = \
            vmap(lambda v: jvp(_forward, (meta_params,), (v,), has_aux=True), in_dims=(in_dims,),
                 out_dims=(None, 0, None), randomness='same')(vs)

        for k in metrics.keys():
            metrics[k] = metrics[k].item()
        # final_text_output = functional_call(
        #     base_lm, updated_params, args=(),
        #     kwargs=dict(input_ids=update_batch['text_ids'], attention_mask=update_batch['text_attention'],
        #                 labels=text_labels)
        # )
        # final_text_loss = weighted_lm_loss_with_outputs(
        #     final_text_output, update_batch['text_ids'], text_labels, weights=None, ignore_index=IGNORED_INDEX
        # )
        # metrics['text_loss'] = final_text_loss.item()
        # metrics['text_gain'] = init_text_loss.item() - final_text_loss.item()

        # del final_text_output
        # torch.cuda.empty_cache()

        # assign grads
        for k, p in self.named_parameters():
            if p.grad is None:
                p.grad = vmap(lambda w, v: w * v / grad_acc_steps, in_dims=(0, 0), out_dims=0)(jvp_output, vs[k]).mean(
                    0)
            else:
                p.grad = vmap(lambda w, v: w * v / grad_acc_steps, in_dims=(0, 0), out_dims=0)(jvp_output, vs[k]).mean(
                    0) + p.grad
        if update:
            # load updated params to base_lm
            for k, p in base_lm.named_parameters():
                p.data = updated_params[k].data

        return total_loss, metrics

    def step(self, update_batch, loc_batches={}, base_lm=None, train=True, sequential_update=False, trgu=False):
        if base_lm is None:
            base_lm = self.base_lm

        text_labels = update_batch['text_ids'].clone()
        text_labels[update_batch['text_attention'] != 1] = -100

        with torch.no_grad():
            batch_size = len(update_batch['text_ids'])
            if batch_size < MAX_BATCH_SIZE:
                init_text_loss = base_lm(input_ids=update_batch['text_ids'],
                                         attention_mask=update_batch['text_attention'],
                                         labels=text_labels).loss
            else:
                leftover = batch_size % MAX_BATCH_SIZE
                init_text_loss = 0
                for i in range(0, batch_size, MAX_BATCH_SIZE):
                    shift = leftover if i + MAX_BATCH_SIZE > batch_size else MAX_BATCH_SIZE
                    init_text_loss += base_lm(input_ids=update_batch['text_ids'][i:i + shift],
                                              attention_mask=update_batch['text_attention'][i:i + shift],
                                              labels=text_labels[i:i + shift]).loss
                    torch.cuda.empty_cache()
                init_text_loss /= (batch_size // MAX_BATCH_SIZE + (leftover > 0))
            init_qa_outputs = base_lm(input_ids=update_batch['qa_ids'], attention_mask=update_batch['qa_attention'],
                                      labels=update_batch['qa_target_ids'])

        updated_lm, weights = self.get_updated_model(update_batch, base_lm, higher_grad=train,
                                                     sequential_update=sequential_update, trgu=trgu)

        updated_lm.eval()
        qa_loss = updated_lm(input_ids=update_batch['qa_ids'], attention_mask=update_batch['qa_attention'],
                             labels=update_batch['qa_target_ids']).loss

        with torch.no_grad():
            if batch_size < MAX_BATCH_SIZE:
                final_text_loss = updated_lm(input_ids=update_batch['text_ids'],
                                             attention_mask=update_batch['text_attention'], labels=text_labels).loss
            else:
                leftover = batch_size % MAX_BATCH_SIZE
                final_text_loss = 0
                for i in range(0, batch_size, MAX_BATCH_SIZE):
                    shift = leftover if i + MAX_BATCH_SIZE > batch_size else MAX_BATCH_SIZE
                    final_text_loss += updated_lm(input_ids=update_batch['text_ids'][i:i + shift],
                                                  attention_mask=update_batch['text_attention'][i:i + shift],
                                                  labels=text_labels[i:i + shift]).loss
                    torch.cuda.empty_cache()
                final_text_loss /= (batch_size // MAX_BATCH_SIZE + (leftover > 0))

        metrics = {'text_loss': final_text_loss.item(),
                   'text_gain': init_text_loss.item() - final_text_loss.item(),
                   'qa_loss': qa_loss.item(),
                   'qa_gain': init_qa_outputs.loss.item() - qa_loss.item()
                   }

        total_loss = qa_loss

        # add additional terms
        if self.c_kl != 0:
            for name, loc_batch in loc_batches.items():
                loc_text_labels = loc_batch['loc_ids'].clone()
                loc_text_labels[loc_batch['loc_attention'] != 1] = -100
                with torch.no_grad():
                    init_loc_qa_outputs = base_lm(input_ids=loc_batch['loc_ids'],
                                                  attention_mask=loc_batch['loc_attention'],
                                                  labels=loc_text_labels)
                post_loc_qa_outputs = updated_lm(input_ids=loc_batch['loc_ids'],
                                                 attention_mask=loc_batch['loc_attention'],
                                                 labels=loc_text_labels)
                kl_div = kl_loc_loss(init_loc_qa_outputs.logits, post_loc_qa_outputs.logits, loc_batch['loc_mask'])
                metrics[f'{name}_kl_div'] = kl_div.item()
                metrics[f'{name}_loc_gain'] = init_loc_qa_outputs.loss.item() - post_loc_qa_outputs.loss.item()
                total_loss += self.c_kl * kl_div / len(loc_batches)

        if (self.norm is not None):
            weight_norm = (((weights ** self.norm) * f.normalize(update_batch['text_attention'][-len(weights):] * 1.,
                                                                 p=1, dim=1)).sum(
                1) ** (1 / self.norm))
            avg_norm = weight_norm.mean()
            weight_norm_from_one = (
                    (((weights - 1) ** self.norm) * f.normalize(update_batch['text_attention'][-len(weights):] * 1.,
                                                                p=1, dim=1)).sum(1) ** (1 / self.norm))
            avg_norm_from_one = weight_norm_from_one.mean()
            # weight_norm = torch.norm(weights, p = self.norm, dim = 1).mean()
            metrics[f'L{self.norm}-norm'] = avg_norm.item()
            metrics[f'L{self.norm}-norm_from_one'] = avg_norm_from_one.item()
            if self.c_norm != 0:
                total_loss += self.c_norm * (avg_norm_from_one if self.norm_from_one else avg_norm)

        metrics['total_loss'] = total_loss.item()
        return total_loss, metrics, updated_lm

    def implicit_step(self, update_batch, loc_batches={}, base_lm=None, sequential_update=True,
                      principle="hessian_free", neumann_steps=10, neumann_alpha=1.0,
                      update=False, grad_acc_steps=1):
        assert sequential_update
        if base_lm is None:
            base_lm = self.base_lm

        text_labels = update_batch['text_ids'].clone()
        text_labels[update_batch['text_attention'] != 1] = -100

        with torch.no_grad():
            init_qa_outputs = base_lm(input_ids=update_batch['qa_ids'], attention_mask=update_batch['qa_attention'],
                                      labels=update_batch['qa_target_ids'])

        updated_lm, weights = self.get_updated_model(update_batch, base_lm, higher_grad=False,
                                                     sequential_update=sequential_update)

        text_labels = update_batch['text_ids'].clone()
        # -100 will trigger vmap error
        text_labels[update_batch['text_attention'] != 1] = IGNORED_INDEX
        qa_labels = update_batch['qa_target_ids'].clone()
        qa_labels[qa_labels == -100] = IGNORED_INDEX

        def _inner_loss_func(_inner_params, _meta_params, _text_ids, _targets, _text_attention, _idx):
            outputs = functional_call(
                base_lm, _inner_params, args=(),
                kwargs=dict(input_ids=_text_ids, attention_mask=_text_attention, labels=_targets)
            )
            _weights = functional_call(self, _meta_params, (_text_ids, _text_attention, _idx))
            loss = weighted_lm_loss_with_outputs(outputs, _text_ids, _targets, _weights, ignore_index=IGNORED_INDEX)
            return loss, outputs

        def _meta_loss_func(_inner_params):
            qa_outputs = functional_call(
                base_lm, _inner_params, args=(),
                kwargs=dict(input_ids=update_batch['qa_ids'], attention_mask=update_batch['qa_attention'],
                            labels=qa_labels)
            )
            # qa_output = updated_lm(input_ids=update_batch['qa_ids'], attention_mask=update_batch['qa_attention'],
            #                           labels=qa_labels)
            qa_loss = weighted_lm_loss_with_outputs(
                qa_outputs, update_batch['qa_ids'], qa_labels, weights=None, ignore_index=IGNORED_INDEX
            )

            metrics = {
                'qa_loss': qa_loss.detach(),
                'qa_gain': init_qa_outputs.loss.detach() - qa_loss.detach()
            }

            total_loss = qa_loss

            if self.c_kl != 0:
                for name, loc_batch in loc_batches.items():
                    loc_text_labels = loc_batch['loc_ids'].clone()
                    loc_text_labels[loc_batch['loc_attention'] != 1] = IGNORED_INDEX
                    init_loc_qa_outputs = base_lm(input_ids=loc_batch['loc_ids'],
                                                  attention_mask=loc_batch['loc_attention'],
                                                  labels=loc_text_labels)
                    post_loc_qa_outputs = functional_call(
                        base_lm, _inner_params, args=(),
                        kwargs=dict(input_ids=loc_batch['loc_ids'], attention_mask=loc_batch['loc_attention'],
                                    labels=loc_text_labels)
                    )
                    kl_div = kl_loc_loss(init_loc_qa_outputs.logits, post_loc_qa_outputs.logits, loc_batch['loc_mask'])
                    metrics[f'{name}_kl_div'] = kl_div.detach()
                    metrics[f'{name}_loc_gain'] = init_loc_qa_outputs.loss.detach() - post_loc_qa_outputs.loss.detach()
                    total_loss += self.c_kl * kl_div / len(loc_batches)

            # omit as self.c_norm is always 0
            # if (self.norm is not None):
            #     weight_norm = (
            #             ((weights ** self.norm) * f.normalize(update_batch['text_attention'] * 1., p=1, dim=1)).sum(
            #                 1) ** (1 / self.norm))
            #     avg_norm = weight_norm.mean()
            #     weight_norm_from_one = ((((weights - 1) ** self.norm) * f.normalize(update_batch['text_attention'] * 1.,
            #                                                                         p=1, dim=1)).sum(1) ** (
            #                                     1 / self.norm))
            #     avg_norm_from_one = weight_norm_from_one.mean()
            #     # weight_norm = torch.norm(weights, p = self.norm, dim = 1).mean()
            #     metrics[f'L{self.norm}-norm'] = avg_norm.detach()
            #     metrics[f'L{self.norm}-norm_from_one'] = avg_norm_from_one.detach()
            #     if self.c_norm != 0:
            #         total_loss += self.c_norm * (avg_norm_from_one if self.norm_from_one else avg_norm)

            metrics['total_loss'] = total_loss.detach()

            return total_loss, metrics

        inner_params = dict(updated_lm.named_parameters())
        meta_params = dict(self.named_parameters())
        # note 1: we use the last inner batch to compute hessian and jacobian
        # note 2: we omit p meta_loss / p meta_params, as c_norm is always set to be 0 in our experiments
        # compute d_T = p meta_loss / p inner_params
        with torch.no_grad():
            d, metrics = grad(_meta_loss_func, has_aux=True)(inner_params)
        # compute d H^{-1}, H = p^2 inner_loss / p inner_params^2
        if principle == "hessian_free":
            v = d
        elif principle == 'neumann':
            def _neumann_forward(_inner_params, i):
                with torch.no_grad():
                    _grad = grad(
                        lambda _inner_params: _inner_loss_func(
                            _inner_params, meta_params, update_batch['text_ids'][i:i + 1], text_labels[i:i + 1],
                            update_batch['text_attention'][i:i + 1], update_batch['idx'][i:i + 1]
                        )[0],
                    )(_inner_params)
                    result = dict((_k, _v - neumann_alpha * _grad[_k]) for _k, _v in _inner_params.items())
                return result
            v = d
            batch_size = len(update_batch['text_ids'])
            for step in range(neumann_steps):
                _, v = jvp(lambda _inner_params: _neumann_forward(_inner_params, step % batch_size), (inner_params,), (v,))
                v = dict((k, v + d[k]) for k, v in v.items())
            v = dict((_k, neumann_alpha * _v) for _k, _v in v.items())
        else:
            raise NotImplementedError()
        # compute d H^{-1} J, J = p^2 inner_loss / p inner_params p meta_params
        (_, vjp_func) = vjp(
            lambda _meta_params: grad(
                lambda _inner_params: _inner_loss_func(
                    _inner_params, _meta_params, update_batch['text_ids'][-1:], text_labels[-1:],
                    update_batch['text_attention'][-1:], update_batch['idx'][-1:]
                )[0],
            )(inner_params),
            meta_params
        )
        g = vjp_func(v)[0]
        # note: minus
        for k, p in self.named_parameters():
            if p.grad is None:
                p.grad = g[k].data * (-1. / grad_acc_steps)
            else:
                p.grad = g[k].data * (-1. / grad_acc_steps) + p.grad
        if update:
            # load updated params to base_lm
            for k, p in base_lm.named_parameters():
                p.data = inner_params[k].data

        return metrics['total_loss'], metrics

    def load(self, epoch=None, checkpoint_step=None, target_path=None):
        """Loads a checkpoint.

        Args:
            either epoch and checkpoint step or an explicit path

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        # we dont need to worry about loading or saving the base_lm because it is assumed to be fixed
        if target_path is None:
            target_path = (
                f'{os.path.join(self._log_dir, "state")}'
                f'{epoch}-{checkpoint_step}.pt'
            )
        if os.path.isfile(target_path):
            state = torch.load(target_path, map_location=torch.device(self.device))
            self.load_state_dict(state['state_dict'], strict=False)
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def save(self, epoch, checkpoint_step, file_name=None):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            epoch (int)
            checkpoint_step (int): iteration to label checkpoint with
        """
        # juggling so we don't have to save base_lm
        temp_base_lm = self.base_lm
        self.base_lm = None
        file_name = file_name or f'state{epoch}-{checkpoint_step}.pt'
        state_dict = self.state_dict()
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(os.path.join(self._log_dir, "checkpoints"), exist_ok=True)
        torch.save(
            dict(state_dict=state_dict), f'{os.path.join(self._log_dir, "checkpoints", file_name)}')

        # juggling so we don't have to save base_lm
        self.base_lm = temp_base_lm
        print('Saved checkpoint.')

    def validate(self, base_lm, val_dataloader, loc_dataloaders={}, reset_base_freq=1, sequential_update=False):
        metrics_dic = defaultdict(lambda: [])
        base_state_dict = {k: v.detach().clone().cpu() for k, v in base_lm.state_dict().items()}
        loc_iters = {k: iter(v) for k, v in loc_dataloaders.items()}

        for i_step, batch in tqdm(enumerate(val_dataloader), desc='validation', position=1, total=len(val_dataloader)):
            loc_batches = {}
            for k in loc_iters.keys():
                try:
                    loc_batches[k] = next(loc_iters[k])
                except StopIteration:
                    loc_iters[k] = iter(loc_dataloaders[k])
                    loc_batches[k] = next(loc_iters[k])

            _, metrics, updated_lm = self.step(batch, loc_batches, base_lm=base_lm, train=False,
                                               sequential_update=sequential_update)

            base_lm.load_state_dict(updated_lm.state_dict())

            if (i_step + 1) % reset_base_freq == 0:
                base_lm.load_state_dict(base_state_dict)
            for k, v in metrics.items():
                metrics_dic[f'[AGG]{k}'].append(v)
                if self.log_stepwise_metrics:
                    metrics_dic[f'[step-{i_step % reset_base_freq}]{k}'].append(v)
        return {k: np.mean(v) for k, v in metrics_dic.items()}

    def plot_weights(self, batch, tokenizer, save_path=None):

        # plot weights for the first text sample in the batch
        l = batch['text_attention'][0].sum().item()
        with torch.no_grad():
            weights = self(batch['text_ids'], batch['text_attention'], idx=batch['idx']).cpu().detach()[0][:l]
        tokens = [tokenizer.decode(t) for t in batch['text_ids'][0]][:l]
        weights = weights[:l]
        max_weight = weights.max()

        image = create_colored_text([f'Max:{max_weight:.3f}, Min:{weights.min():.3f}   :'] + tokens,
                                    [0] + weights.tolist())
        if save_path is not None:
            if save_path[-4:] != '.':
                save_path += '.png'
            image.save(save_path)
        return image

    def old_plot_weights(self, batch, tokenizer, save_path):
        window = 80
        with torch.no_grad():
            weights = self(batch['text_ids'], batch['text_attention'], idx=batch['idx']).cpu().detach()[0]
        l = batch['text_attention'][0].sum().item()
        tokens = [tokenizer.decode(t) for t in batch['text_ids'][0]][:l]
        n_frames = (l - 1) // window + 1
        fig, ax = plt.subplots(n_frames, 1, figsize=(16, 4 * n_frames))
        for i in range(n_frames):
            start = i * window
            end = min(l, start + window)
            my_xticks = tokens[start:end]
            ax[i].set_xticks(range(start, end), my_xticks, rotation=85)
            ax[i].bar(range(start, end), weights[start:end])
        plt.subplots_adjust(hspace=1)
        plt.savefig(save_path, facecolor='white')


class CaMeLSWeightModel(WeightingModel):
    def __init__(self, config=default_config, device_=None):
        if device_ is None:
            device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(config, device_)
        self.gpt_base = GPT2Model.from_pretrained(config.pretrained_model).to(device_)
        self.freeze_base = config.freeze_base
        self.normalize = config.normalize

        non_linearities = {
            None: None,
            'sigmoid': torch.sigmoid,
            'exp': torch.exp,
            'softplus': nn.Softplus(beta=3),
            'gelu': nn.GELU()
        }
        self.nl = non_linearities[config.non_linearity]

        self.fc1 = nn.Linear(self.gpt_base.embed_dim, 128, device=device_)
        self.fc2 = nn.Linear(128, 1, device=device_)
        # initializing the second layer to all 0's makes sense to me i think
        with torch.no_grad():
            self.fc2.weight.fill_(0)
            self.fc2.bias.fill_(0)

    def get_optimizer(self, outer_lr):
        if self.freeze_base:
            return torch.optim.Adam(
                list(self.fc1.parameters()) + list(self.fc2.parameters()),
                lr=outer_lr
            )
        else:
            return torch.optim.Adam(
                self.parameters(),
                lr=outer_lr
            )

    def forward(self, x, attention_mask=None, idx=None):
        # x: [n_batch, n_tokens]
        # x:[ hello bob PAD PAD PAD] attn_mask: [1 1  0 0 0]
        gelu = nn.GELU()
        if attention_mask is None:
            attention_mask = torch.ones(x.shape, device=x.device)
        if self.freeze_base:
            with torch.no_grad():
                x = self.gpt_base(x.to(self.device))['last_hidden_state']  # is there a more elagant way to do this
        else:
            x = self.gpt_base(x)['last_hidden_state']
        x = gelu(self.fc1(x))
        x = self.fc2(x).squeeze()

        if self.nl is not None:
            x = self.nl(x)  # this feels more natural

        x = (x * attention_mask)

        if self.normalize:
            x = x / (x.sum(1).unsqueeze(1)) * (attention_mask.sum(1).unsqueeze(1))
        return x


class UniformWeightModel(WeightingModel):
    def __init__(self, config=default_config, device_=None):
        if device_ is None:
            device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(config, device_)
        self.scalar = torch.tensor(config.init_scalar, dtype=float, requires_grad=True)

    def get_optimizer(self, outer_lr):
        return torch.optim.Adam([self.scalar],
                                lr=outer_lr
                                )

    def validate(self, val_dataloader, loc_dataloader=None):
        metrics = super().validate(val_dataloader, loc_dataloader=loc_dataloader)
        metrics['scalar'] = self.scalar.item()
        return metrics

    def load(self, epoch=None, checkpoint_step=None, target_path=None):
        pass

    # dummy model that just rescales uniform weights
    def forward(self, x=None, attention_mask=None, idx=None):
        return self.scalar * attention_mask


class SSM(WeightingModel):
    def __init__(self, tokenizer='gpt2', device_=None,
                 entities_to_ignore=['TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']):
        self.nlp = spacy.load("en_core_web_sm")
        super().__init__(device_=device_)
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.entities_to_ignore = entities_to_ignore

    def forward(self, x, attention_mask, idx=None):
        batch_weights = []
        for i in range(len(x)):
            named_ents = get_nes_from_toks(x[i][attention_mask[i] == 1], self.tokenizer, self.nlp,
                                           entities_to_ignore=self.entities_to_ignore)
            padding = [0] * (len(x[i]) - len(named_ents))
            weights = np.concatenate((named_ents, padding))
            batch_weights.append(weights)
        return torch.tensor(batch_weights).to(self.device)


class TFIDF(WeightingModel):
    def __init__(self, dataset, dataset_args, tokenizer='gpt2', device_=None, min_threshold=None):
        super().__init__(device_=device_)
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        if dataset == 'streamingqa':
            self.fit_data = list(pd.read_csv(dataset_args['streamingqa_path'])['text'].unique())
        elif dataset == 'squad':
            squad_ds = load_dataset('squad')
            self.fit_data = []
            for split in dataset_args['squad_splits']:
                self.fit_data += list(squad_ds[split]['context'])
            self.fit_data = list(set(self.fit_data))

        elif dataset == 'archivalqa':
            self.fit_data = list(pd.read_csv(dataset_args['archivalqa_path'])['ans_paragraph'].unique())
        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(self.fit_data)
        self.doc_tokenizer = self.tfidf.build_analyzer()
        self.doc_preprocesser = self.tfidf.build_preprocessor()
        self.feature_names = self.tfidf.get_feature_names_out()
        if min_threshold is not None:
            arr = self.tfidf.transform(self.fit_data).toarray()
            non_zero_tfidf = arr[arr != 0]
            self.min_thresh = np.quantile(non_zero_tfidf, min_threshold)
        else:
            self.min_thresh = 0

    def forward(self, x, attention_mask, idx=None):
        batch_weights = []
        for i in range(len(x)):
            tfidf_weights = self.process_single_seq(x[i][attention_mask[i] == 1])
            padding = [0] * (len(x[i]) - len(tfidf_weights))
            weights = np.concatenate((tfidf_weights, padding))
            batch_weights.append(weights)
        batch_weights = torch.tensor(batch_weights).to(self.device)
        return batch_weights * (batch_weights > self.min_thresh)

    def process_single_seq(self, tokens):
        # reconstruct text from tokens
        text_by_toks = [self.tokenizer.decode(t, clean_up_tokenization_spaces=False) for t in tokens]
        text = ''.join(text_by_toks)
        # apply TFIDF to text
        pre_processed_doc = self.doc_preprocesser(text)
        assert len(pre_processed_doc) == len(text)
        word_splits = self.doc_tokenizer(pre_processed_doc)
        results = self.tfidf.transform([pre_processed_doc]).toarray()
        # convert from sparse matrix to dict
        word_to_tfidf = {}
        for j, term in enumerate(self.feature_names):
            tfidf_value = results[0, j]
            if tfidf_value > 0:
                word_to_tfidf[term] = tfidf_value

        # map TFIDF to characters
        per_char_tf_idf = np.zeros(len(pre_processed_doc))
        cur_idx = 0
        for word in word_splits:
            word_start = pre_processed_doc.find(word, cur_idx)
            new_end = word_start + len(word)
            if word in word_to_tfidf:
                per_char_tf_idf[word_start:new_end] = word_to_tfidf[word]
            cur_idx = new_end

        tok_lens = [len(t) for t in text_by_toks]
        prefix_sum = np.cumsum(tok_lens, dtype=np.int32)
        return [max(per_char_tf_idf[prefix_len - tok_len:prefix_len]) for tok_len, prefix_len in
                zip(tok_lens, prefix_sum)]


def rademacher_like(x, n, device):
    if isinstance(x, tuple):
        return tuple([torch.randint(0, 2, (n, *xx.shape), dtype=torch.float32, device=device) * 2. - 1. for xx in x])
    elif isinstance(x, dict):
        return dict(
            (k, torch.randint(0, 2, (n, *v.shape), dtype=torch.float32, device=device) * 2. - 1.) for k, v in x.items())
    else:
        raise ValueError('Unknown type {} for rademacher_like'.format(type(x)))
