def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''

    parser.add_argument('--n_episodes', type=int, default=10000000000, help='the number of episodes to train for')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training')
    parser.add_argument('--memory_size', type=int, default=100, help='size of replay memory')
    parser.add_argument('--gamma', type=float, default=0.99, help='the gamma value')
    parser.add_argument('--render', type=bool, default=False, help='whether to render the env')
    parser.add_argument('--target_update_int', type=int, default=1000, help='the target update interval')
    return parser
