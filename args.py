import argparse

def load_hyperparameters():
	parser = argparse.ArgumentParser(description='Reinforcement Learning for Distant Supervision Relation Extraction')
	parser.add_argument('-max_sent', type=int, default=120, help='maximum length of sentence')
	parser.add_argument('-max_pos', type=int, default=100, help='maximum distance of position feature')
	parser.add_argument('-pos_num', type=int, default=100*2+2, help='the dimension of position matrix')

	parser.add_argument('-hidden_size', type=int, default=100, help='hidden size of policy cnn network')
	parser.add_argument('-RC_hidden_size', type=int, default=100, help='hidden size of relation classifier cnn network')
	parser.add_argument('-window_size', type=list, default=[3], help='window size of policy cnn network, and support multi window size')
	parser.add_argument('-word_dim', type=int, default=50, help='the dimension of word embeddings')
	parser.add_argument('-pos_dim', type=int, default=5, help='the dimension of position embeddings')
	parser.add_argument('-label_num', type=int, default=2, help='label number')

	parser.add_argument('-log_interval', type=int, default=100, help='how many steps to wait before logging training status')
	parser.add_argument('-batch_size', type=int, default=160, help='batch size')
	parser.add_argument('-max_epoch', type=int, default=300, help='the maximum training epoch')
	parser.add_argument('-learning_rate', type=float, default=0.00002, help='learning rate of reinforcement learning')
	parser.add_argument('-dropout_rate', type=float, default=0.5, help='dropout rate')
	parser.add_argument('-reward_scale', type=float, default=100, help='reward scale')

	parser.add_argument('-model_dir', type=str, default='./model/', help='the learned policy network models')
	parser.add_argument('-cleaned_data_dir', type=str, default='./cleaned_data/', help='the directory of the cleaned data')

	parser.add_argument('-seed', type=int, default=0, help='random seed')
	parser.add_argument('-redistrubute', action='store_true')
	parser.add_argument('-train_version', type=int, default=522611, help='[522611, 570088]')
	parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 means cpu [default: 0]')
	args = parser.parse_args()

	args.save_path = 'models/hyperparam'

	print("------HYPERPARAMETERS-------")
	for k, v in vars(args).items():
		print(k + ': ' + str(v))
	print("----------------------------\n")

	return args
