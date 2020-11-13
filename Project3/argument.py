def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''

    parser.add_argument('--n_episodes', type=int, default=50000, help='the number of episodes to train for')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training')
    parser.add_argument('--memory_size', type=int, default=1300, help='size of replay memory')
    parser.add_argument('--continue_training', action='store_true', default=False,
                        help='Whether to continue training from an existing model')
    parser.add_argument('--target_update_int', type=int, default=10000, help='the target update interval in steps')
    parser.add_argument('--save_interval', type=int, default=10000, help='the save interval in steps')
    parser.add_argument('--m_load_path', type=str, default="./saved_models/model.pth", help='the model load path for testing')
    parser.add_argument('--m_save_path', type=str, default="./saved_models/model-latest.pth",
                        help='the model save path for training')
    parser.add_argument('--l_save_path', type=str, default="./saved_models/model-latest-log.csv",
                        help='the log save path')
    return parser
