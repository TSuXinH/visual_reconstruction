import os

sig = [True, False]
lr = [1e-4, 5e-4, 1e-3, 5e-3]
ep = [500, 1000]
n_latents = [64, 128, 256, 512]

i = 0
for a in sig:
    for b in lr:
        for c in ep:
            for d in n_latents:
                i += 1
                if i <= 3:
                    continue
                lr_repr = '1e-4' if lr == 1e-4 else '5e-4' if lr == 5e-4 else '1e-3' if lr == 1e-3 else '5e-3'
                print('start training --with_sigmoid {} --max_epoch {} --n_ae_latents {} --lr {}'.format(a, c, d, lr_repr))
                os.system('python x_train.py --with_sigmoid {} --max_epoch {} --n_ae_latents {} --lr {}'.format(a, c, d, lr_repr))
                print('end training --with_sigmoid {} --max_epoch {} --n_ae_latents {} --lr {}'.format(a, c, d, lr_repr))
