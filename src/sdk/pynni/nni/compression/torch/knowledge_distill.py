import copy
import logging
import torch
import torch.nn.functional as F

_logger = logging.getLogger(__name__)


class KnowledgeDistill():
    def __init__(self, teacher_model, kd_T, kd_beta):
        self.teacher_model = teacher_model
        self.kd_T = kd_T
        self.kd_beta = kd_beta

    def get_kd_loss(self, data, student_out):
        with torch.no_grad():
            kd_out = self.teacher_model(data)
        soft_log_out = F.log_softmax(student_out / self.kd_T, dim=1)
        soft_t = F.softmax(kd_out / self.kd_T, dim=1)
        loss_kd = F.kl_div(soft_log_out, soft_t.detach(), reduction='batchmean')
        return loss_kd
