ESC_10 = False
ESC_50 = False	
US8K = False
ADSMI = True

path_to_ESC50 = './data/ESC50'
path_to_ESC10 = './data/ESC10'
path_to_US8K = './data/US8K'

path_to_classifierModel = './data/results/2020-12-22-10-42/'


ESC10_classIds = [0, 1, 10, 11, 12, 20, 21, 38, 40, 41]


if ESC_50:
	class_numbers = 50
else:
	class_numbers = 10


if ESC_10 or ESC_50:
	lr = 5e-4 #for ESC-50 and ESC-10
	folds = 5
	test_fold = [1]
	train_folds = list(i for i in range(1, 6) if i != test_fold[0])	
else:
	lr = 1e-4 # for US8K
	fold = 10
	test_fold = [1]
	train_folds = list(i for i in range(1, 11) if i != test_fold[0])	
	
if US8K:
	us8k_train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
	us8k_test_fold = [9]

if ADSMI:
	ADSMI_train_folds = [1, 2, 3, 4, 5, 6, 7]
	ADSMI_test_fold = [8]
	lr = 1e-4
	class_numbers = 4

temperature = 0.05
alpha = 0.5

freq_masks = 2
time_masks = 1
freq_masks_width = 16
time_masks_width = 16

epochs = 300
finetune_epochs = 400
patience = 120
batch_size = 64
warm_epochs = 10
gamma = 0.98

#Resnet parameters
channels = 3
#pipeline
desired_length_in_seconds = 10