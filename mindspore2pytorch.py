from mindspore.train.serialization import load_checkpoint
import torch


def mindspore2pytorch(ckpt_name='D:\\python_project\\Categroy-relate attention alignment\\weights\\ghostnetv2_1x.ckpt'):
    par_dict = load_checkpoint(ckpt_name)
    state_dict = {}
    state_dict_useless = ['global_step', 'learning_rate', 'beta1_power', 'beta2_power']

    for name in par_dict:
        parameter = par_dict[name].data

        if name in state_dict_useless or name.startswith('moment1.') or name.startswith('moment2.'):
            pass

        else:
            print('========================ms_name:', name)

            if name.endswith('.beta'):
                name = name[:name.rfind('.beta')]
                name = name + '.bias'

            elif name.endswith('.gamma'):
                name = name[:name.rfind('.gamma')]
                name = name + '.weight'

            elif name.endswith('.moving_mean'):
                name = name[:name.rfind('.moving_mean')]
                name = name + '.running_mean'

            elif name.endswith('.moving_variance'):
                name = name[:name.rfind('.moving_variance')]
                name = name + '.running_var'

            print('========================py_name:', name)
            state_dict[name] = torch.from_numpy(parameter.asnumpy())  ###


    torch.save({'state_dict': state_dict}, 'ghostv2.pth')

if __name__ == '__main__':
    mindspore2pytorch()