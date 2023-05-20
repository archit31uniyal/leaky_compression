from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, get_vic_adv_preds_on_distr
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.config import DatasetConfig, AttackConfig, BlackBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult
from distribution_inference.models.core import MyAlexNet, ResNet18, MLPThreeLayer
from distribution_inference.nleaked.nleaked import BinaryRatio
from pretrain import compute_accuracy
from CKA import CKA
from attack_utils import *
# from torch_cka import CKA

torch.manual_seed(3407)

def JI_attack(M_target, M_shadow_M0, M_shadow_M1):
    """
    JI_threshold = 0.3679
    """
    # threshold = np.exp(-1)
    ji_M0, _, _ = calculate_JI(M_target, M_shadow_M0, 'conv', 1)
    ji_M1, _, _ = calculate_JI(M_target, M_shadow_M1, 'conv', 1)


    ji_target_M0 = get_JI(ji_M0)
    ji_target_M1 = get_JI(ji_M1)

    # print(ji_target_M0, ji_target_M1, "\n")

    if ji_target_M0 > ji_target_M1:
        result = 0
    else:
        result = 1
    # JI_exp = np.exp(-(ji_target_M0/ji_target_M1))
    # if JI_exp < threshold:
    #     result = 1
    # else:
    #     result = 0
    # print(ji_target_M0, ji_target_M1)
    return result

def l_attack(M_target, M0_shadow, M1_shadow, M_pretrain, d_aux, pretrain = False):
    M_target.cuda()
    M0_shadow.cuda()
    loss_target = get_loss_per_sample(M_target, M_pretrain, d_aux, pretrain)
    l2_target = np.linalg.norm(loss_target, 2)
    loss_M0 = get_loss_per_sample(M0_shadow, M_pretrain, d_aux, pretrain)
    l2_M0 = np.linalg.norm(loss_M0, 2)
    M0_shadow.cpu()
    
    M1_shadow.cuda()
    loss_M1 = get_loss_per_sample(M1_shadow, M_pretrain, d_aux, pretrain)
    l2_M1 = np.linalg.norm(loss_M1, 2)
    M1_shadow.cpu()

    if abs(l2_target- l2_M0) <= abs(l2_target- l2_M1):
        return 0
    else:
        return 1
    # print(f"JI: {ji_coefficient}")
    
def cka_attack(M_target, M0, M1, d_aux):
    cka_0 = CKA(M_target, M0, model1_name='victim', model2_name='0.5M', device= 'cuda')
    # M_target.cpu()
    # M0.cpu()
    results_0_D0 = cka_0.compare(d_aux)
    M0.cpu()
    torch.cuda.empty_cache()

    # gc.collect()
    # torch.cuda.reset_current_context()

    cka_1 = CKA(M_target, M1, model1_name='victim', model2_name='0.8M', device= 'cuda')
    # model1_layers=layers, model2_layers=layers, 
    results_1_D0 = cka_1.compare(d_aux)
    M_target.cpu()
    M1.cpu()
    torch.cuda.empty_cache()
    # gc.collect()
    # torch.cuda.reset_current_context() 

    mask = torch.diag(torch.tensor([1]* results_0_D0['CKA'].shape[0]))
    cka_0 = results_0_D0['CKA'] * mask
    cka_1 = results_1_D0['CKA'] * mask

    cka_mean_0_D0 = torch.diagonal(cka_0, 0).mean()
    cka_mean_1_D0 = torch.diagonal(cka_1, 0).mean()

    if cka_mean_0_D0 >= cka_mean_1_D0:
        return 0
    else:
        return 1

def loss_thresh(M_target, M_p, threshold_0, frac_threshold, d1, points, alpha, pretrain = True):
    M_target.cuda()
    result = inference_per_sample(M_target, M_p, d1, points, threshold_0, alpha, pretrain)
    fraction = np.mean(result)
    M_target.cpu()
    # print(result.count(0), result.count(1),"\n")
    if fraction >= frac_threshold:
        return 1, fraction 
    else: 
        return 0, fraction
    # return result

def loss_attack_per_sample(M_target, M_p, d1, avg_loss, points_0, points_1, pretrain = True):
    M_target.cuda()
    assert (any(points_0 != points_1))
    loss_0 = get_loss_per_data_sample(M_target, M_p, d1, points_0, pretrain)
    loss_1 = get_loss_per_data_sample(M_target, M_p, d1, points_1, pretrain)
    M_target.cpu()
    assert ((np.mean(np.array(loss_0) - avg_loss.item()) <= np.mean(np.array(loss_1) - avg_loss.item())) == (np.mean(np.array(loss_0)) <= np.mean(np.array(loss_1))))
    if np.mean(np.array(loss_0) - avg_loss.item()) <= np.mean(np.array(loss_1) - avg_loss.item()):
        return 0
    else:
        return 1
    
    # vote = [0 if l0 <= l1 else 1 for l0, l1 in zip(loss_0, loss_1)]
    # return vote, p0, p1
    # print(result.count(0), result.count(1),"\n")

def main():
    global write_path, pretrain_model, layers
    BATCH_SIZE = 1
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--en", help="experiment name",
        type=str, required=True)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument('--gpu',
                        default='0,1,2,3', help="device number")
    parser.add_argument(
        "--victim_path", help="path to victim'smodels directory",
        type=str, default=None)
    parser.add_argument(
        "--prop", help="Property for which to run the attack",
        type=str, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)

    # Use given prop (if given) or the one in the config
    if args.prop is not None:
        attack_config.train_config.data_config.prop = args.prop

    # Extract configuration information from config file
    bb_attack_config: BlackBoxAttackConfig = attack_config.black_box
    train_config: TrainConfig = attack_config.train_config
    data_config: DatasetConfig = train_config.data_config
    if train_config.misc_config is not None:
        # TODO: Figure out best place to have this logic in the module
        if train_config.misc_config.adv_config:
            # Scale epsilon by 255 if requested
            if train_config.misc_config.adv_config.scale_by_255:
                train_config.misc_config.adv_config.epsilon /= 255

    # Print out arguments
    flash_utils(attack_config)
    # Define logger
    logger = AttackResult(args.en, attack_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    p1, p2 = 1.0, 1.0

    data_config_adv_1, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config, prop_value = p1)
    
    data_config_adv_2, data_config_vic_2 = get_dfs_for_victim_and_adv(
        data_config, prop_value = p2)
    
    # ds_vic_1 = ds_wrapper_class(
    #     data_config_vic_1,
    #     skip_data=True,
    #     label_noise=train_config.label_noise,
    #     epoch=attack_config.train_config.save_every_epoch)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1)

    ds_adv_2 = ds_wrapper_class(data_config_adv_2)
    _, d_aux_1 = ds_adv_1.get_loaders(batch_size=32, eval_shuffle = False, val_factor = 1)
    _, d_aux_2 = ds_adv_2.get_loaders(batch_size=BATCH_SIZE, eval_shuffle = False, val_factor = 1)

    # d = torch.utils.data.DataLoader(d_aux_2.dataset[:64], batch_size= 64, shuffle = False)


    write_path = f'/p/compressionleakage/logs/Compressed/compression_cv/log_reports/JI_scores_celeba.csv'

    algo = 'autocompress'
    dataset_name = 'celeba'
    # classname = 6

    pretrain_model = pretrain(True, dataset_name)

    M_target = copy.deepcopy(pretrain_model)
    vic_models = [copy.deepcopy(M_target) for _ in range(40)]
    # vic_models_5 = [copy.deepcopy(M_target) for _ in range(10)]
    # vic_models_8 = [copy.deepcopy(M_target) for _ in range(10)]
    adv_models_5_M = [copy.deepcopy(M_target) for _ in range(90)]
    adv_models_8_M = [copy.deepcopy(M_target) for _ in range(90)]

    prop_1, prop_2 = 0.2, 0.5
    br = BinaryRatio(prop_1, prop_2)

    path_compressed_vic5 = f'/p/compressionleakage/logs/Compressed/compression_cv/models/celeba/resnet18/victim/Male/{prop_1}/ft_train_0.5000'
    path_compressed_vic8 = f'/p/compressionleakage/logs/Compressed/compression_cv/models/celeba/resnet18/victim/Male/{prop_2}/ft_train_0.5000'
    path_compressed_adv5 = f'/p/compressionleakage/logs/Compressed/compression_cv/models/celeba/resnet18/adv/Male/{prop_1}/ft_train_0.5000'
    path_compressed_adv8 = f'/p/compressionleakage/logs/Compressed/compression_cv/models/celeba/resnet18/adv/Male/{prop_2}/ft_train_0.5000'

    path_vic5 = '/p/compressionleakage/logs/Compressed/compression_cv/models/celeba/resnet18_new/victim/Male/0.5'
    path_vic8 = '/p/compressionleakage/logs/Compressed/compression_cv/models/celeba/resnet18_new/victim/Male/0.8'
    path_adv5 = '/p/compressionleakage/logs/Compressed/compression_cv/models/celeba/resnet18_new/adv/Male/0.5'
    path_adv8 = '/p/compressionleakage/logs/Compressed/compression_cv/models/celeba/resnet18_new/adv/Male/0.8'

    j = 0
    for i in tqdm(os.listdir(path_compressed_vic5), desc = f"Fetching victim models {prop_1} male"):
        vic_models[j].load_state_dict(torch.load(f'{path_compressed_vic5}/{i}', map_location= "cpu"))
        # vic_models[j] = torch.compile(vic_models[j])
        j += 1
    
    for i in tqdm(os.listdir(path_compressed_vic8), desc = f"Fetching victim models {prop_2} male"):
        vic_models[j].load_state_dict(torch.load(f'{path_compressed_vic8}/{i}', map_location= "cpu"))
        # vic_models[j] = torch.compile(vic_models[j])
        j += 1
 
    j = 0
    for i in tqdm(os.listdir(path_compressed_adv5), desc = f"Fetching adversary models {prop_1} male"):
        adv_models_5_M[j].load_state_dict(torch.load(f'{path_compressed_adv5}/{i}', map_location= "cpu"))
        # adv_models_5_M[j] = torch.compile(adv_models_5_M[j])
        j += 1
    
    j = 0
    for i in tqdm(os.listdir(path_compressed_adv8), desc = f"Fetching adversary models {prop_2} male"):
        adv_models_8_M[j].load_state_dict(torch.load(f'{path_compressed_adv8}/{i}', map_location= "cpu"))
        # adv_models_8_M[j] = torch.compile(adv_models_8_M[j])
        j += 1


    result_JI = []
    result = []
    # result_loss = []
    vic_fractions = []
    vic_models_ = vic_models[:15] + vic_models[20:35]
    vic_progress = tqdm(range(len(vic_models_))) 

    p = True
    threshold = get_thresholds(adv_models_5_M[:30], adv_models_8_M[:30], pretrain_model, d_aux_2, p)
    
    mean_0 = threshold[:][1]
    mean_1 = threshold[:][2]
    # index = min(len(np.where((mean_0-mean_1) <= -0.5)[0]), len(np.where((mean_0-mean_1) >= 0.5)[0]))
    index = int(0.05 * len(mean_0))
    ordering = threshold[:][-1]
    points_0 = ordering[:index]
    points_1 = ordering[-index:]
    threshold = threshold[:][0]

    get_plot(range(len(mean_0)), y = (mean_0-mean_1), y1 = mean_0, y2 = mean_1, scatter= True, title=None, xlabel = None, ylabel="Mean loss of shadow models per image", save_path=f"loss_adv_60_{prop_1*10}_{prop_2*10}_data_{p2*100}.png", pdf = True, label_0 = f'Distribution {prop_1 *100} males', label_1= f'Distribution {prop_2 *100} males')

    # exit()
    
    result_JI = []
    for i in vic_progress:
        avg_loss = get_loss_pretrain(M_target, pretrain_model, d_aux_2)
        res = loss_attack_per_sample(vic_models_[i], pretrain_model, d_aux_2, avg_loss, points_0, points_1, p)
        result_JI.append(res)
    
        vic_progress.set_description(f'Victim {i+1} | Avg_loss: {avg_loss} | Inference: {result_JI}')

    result.append(round((result_JI[:15].count(0) + result_JI[15:].count(1))*100/len(result_JI), 2))
    acc = round((result_JI[:15].count(0) + result_JI[15:].count(1))*100/len(result_JI), 2)
    print(f"Attack accuracy (n_leaked): {np.mean(result)} ({br.get_n_effective(acc/100)}) \n Std Dev: {np.std(result)}")
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()