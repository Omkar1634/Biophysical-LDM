import json, os

# data_path = r'D:\Github\PhD Code\FFHQ-UV\dataset'

# for root, dirs, files in os.walk(data_path):
#     level = root.replace(data_path, '').count(os.sep)
#     indent = '  ' * level
#     print(f'{indent}{os.path.basename(root)}/')
#     if level < 2:
#         for f in files[:5]:
#             print(f'{indent}  {f}')
#         if len(files) > 5:
#             print(f'{indent}  ... ({len(files)} total files)')


base = r'D:\Github\PhD Code\FFHQ-UV\ldm_pretrain'
for name, path in [
    ('UNET',    'unet/config.json'),
    ('VAE',     'vae/config.json'),
    ('TEXT_ENC','text_encoder/config.json'),
    ('SCHED',   'scheduler/scheduler_config.json'),
]:
    full = os.path.join(base, path)
    with open(full) as f:
        d = json.load(f)
    print(f'\n── {name} ──')
    for k,v in d.items():
        print(f'  {k}: {v}')