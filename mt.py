import torch
import torch.distributed as dist
from collections import OrderedDict


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

class ATeacherTrainer():
    def __init__(self,keep_rate=0.996):
        self.keep_rate = keep_rate

    @torch.no_grad()
    def _copy_main_model(self,model,model_teacher):
        # initialize all parameters
        if get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in model.state_dict().items()
            }
            new_model = OrderedDict()
            for key, value in rename_model_dict.items():
                if key in model_teacher.keys():
                    new_model[key] = rename_model_dict[key]
            model_teacher.load_state_dict(new_model)
            # model_teacher.load_state_dict(rename_model_dict)
        else:
            new_model = OrderedDict()
            for key, value in model.state_dict().items():
                if key in model_teacher.state_dict().keys():
                    new_model[key] = value
            model_teacher.load_state_dict(new_model)
        return model_teacher,model

    @torch.no_grad()
    def _update_teacher_model(self, model,model_teacher):
        if get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in model.state_dict().items()
            }
        else:
            student_model_dict = model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - self.keep_rate) + value * self.keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        model_teacher.load_state_dict(new_teacher_dict)
        return model_teacher, model