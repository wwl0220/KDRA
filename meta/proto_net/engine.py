import torch.optim as optim


class Engine(object):
    def __init__(self):
        hook_names = ['on_start', 'on_start_epoch', 'on_sample', 'on_forward',
                      'on_backward', 'on_end_epoch', 'on_update', 'on_end']

        self.hooks = {}
        for hook_name in hook_names:
            self.hooks[hook_name] = lambda state: None

    def train(self, model, train_data, val_data, max_epoch, train_flag, start_time, is_cuda):
        state = {
            'model': model,
            'train_data': train_data,
            'val_data': val_data,
            'max_epoch': max_epoch,
            'train_flag': train_flag,
            'epoch': 0,
            'best_model_epoch': 0,
            't': 0,
            'batch': 0,
            'stop': False,
            'start_time': start_time,
            'is_cuda': is_cuda
        }

        state['optimizer'] = optim.Adam(state['model'].parameters(), lr=0.01, weight_decay=0.001)

        self.hooks['on_start'](state)
        while state['epoch'] < state['max_epoch'] and not state['stop']:
            state['model'].train()

            self.hooks['on_start_epoch'](state)

            state['episodes'] = len(state['train_data'])

            for sample in state['train_data']:
                state['sample'] = sample
                self.hooks['on_sample'](state)

                state['proto'], loss, state['output'] = state['model'].loss(state['sample'], state['is_cuda'])
                self.hooks['on_forward'](state)

                state['optimizer'].zero_grad()
                loss.backward()
                self.hooks['on_backward'](state)
                state['optimizer'].step()

                state['scheduler'].step()  # pytorch 1.3.1

                state['t'] += 1
                state['batch'] += 1
                self.hooks['on_update'](state)

            state['epoch'] += 1
            state['batch'] = 0
            self.hooks['on_end_epoch'](state)

        self.hooks['on_end'](state)

        return state['best_model_epoch'], state['best_model_time']
