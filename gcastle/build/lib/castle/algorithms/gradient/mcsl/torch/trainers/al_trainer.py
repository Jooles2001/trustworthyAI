# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numpy as np
import torch

from ..helpers.utils import compute_acyclicity, convert_logits_to_sigmoid, callback_after_training
from castle.common.consts import LOG_FREQUENCY, LOG_FORMAT


logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
# logger = logging.getLogger(__name__)  # @Jules 11/07/2023: manually change "logging" to "logger"

class Trainer(object):
    """
    Augmented Lagrangian method with first-order gradient-based optimization
    """

    def __init__(self, model, learning_rate, init_rho, rho_thresh, h_thresh,
                 rho_multiply, init_iter, h_tol, temperature,
                 device=None) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.init_rho = torch.tensor(init_rho, device=device)
        self.rho_thresh = rho_thresh
        self.h_thresh = h_thresh
        self.rho_multiply = rho_multiply
        self.init_iter = init_iter
        self.h_tol = h_tol
        self.temperature = torch.tensor(temperature, device=device)
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate,)

    def train(self, x, max_iter, iter_step) -> torch.Tensor:
        """"""

        rho = self.init_rho
        h = np.inf
        h_new = np.inf
        alpha = 0.0
        w_logits_new = None
        for i in range(1, max_iter + 1):
            logging.info(f'Current epoch: {i}==================')
            print("Changes made by Jules")
            # @Jules 11/07/2023: Additional loggings
            logging.info(f'Current rho: {rho}==================') # @Jules 11/07/2023 : rho is the penalty parameter ?
            logging.info(f'Current adjacency matrix: \n {callback_after_training(w_logits_new, self.temperature, 0.5)}==================') # @Jules 11/07/2023 : w_logits_new is the adjacency matrix
            while rho < self.rho_thresh:
                loss_new, h_new, w_logits_new = self.train_step(
                    x, iter_step, rho, alpha, self.temperature
                )
                if h_new > self.h_thresh * h:
                    rho *= self.rho_multiply
                else:
                    break
            # Use two stopping criterion
            h_logits = compute_acyclicity(
                convert_logits_to_sigmoid(w_logits_new.detach(),
                                          tau=self.temperature)
            )
            logging.info(f'Current        h: {h_new}')
            logging.info(f'Current h_logits: {h_logits}')
            if (h_new <= self.h_tol
                    # and h_logits <= self.h_tol
                    and i > self.init_iter):
                break
            # Update h and alpha
            h = h_new.detach().cpu()
            alpha += rho * h

        return w_logits_new

    def train_step(self, x, iter_step, rho, alpha, temperature) -> tuple:

        # curr_loss, curr_mse and curr_h are single-sample estimation
        curr_loss, curr_h, curr_w = None, None, None
        for _ in range(iter_step):

            (curr_loss, curr_h, curr_w) = self.model(x, rho, alpha, temperature)

            self.optimizer.zero_grad()
            curr_loss.backward()
            self.optimizer.step()

            if _ % LOG_FREQUENCY == 0:
                logging.info(f'Current loss in step {_}: {curr_loss.detach()}')

        return curr_loss, curr_h, curr_w
