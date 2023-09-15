    for command in delete_incomplete launch
    do
    python -m domainbed.scripts.sweep ${command}  --data_dir /home/zzq/data --skip_confirmation --single_test_envs\
    --output_dir=./domainbed/output/MMD/Res18bc/  --command_launcher local --algorithms MMD \
    --datasets PACS  --n_hparams 1 --n_trials 1 --hparams """{\"batch_size\":64,\"lr\":1e-05,\"resnet18\":1,\"resnet_dropout\":0.0,\"weight_decay\":0.0}""" 
    #n_params 随机种子的个数，从0开始  n_trials dataset不同split 的个数
    # --hparams """{\"batch_size\":64,\"lr\":1e-05,\"resnet_dropout\":0.0,\"weight_decay\":0.0}"""
    done


    #     parser.add_argument('--hparams_seed', type=int, default=0,
    #     help='Seed for random hparams (0 means "default hparams")')
    # parser.add_argument('--trial_seed', type=int, default=0,
    #     help='Trial number (used for seeding split_dataset and '
    #     'random_hparams).')
    # parser.add_argument('--seed', type=int, default=0,
    #     help='Seed for everything else')