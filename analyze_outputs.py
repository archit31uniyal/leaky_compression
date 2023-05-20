import torch
import pickle
from PIL import Image
import csv
from compression_attack_cv import *

dataset_name = "celeba"
model = pretrain(True, dataset_name)

def PNormDist(x1, x2, p =2):
    if len(x1.shape) == 1 and len(x2.shape) == 1:
        x = torch.cdist([x1], [x2], p)
    else:     
        x = torch.cdist(x1, x2, p)
    return x

def dist_bn(x1, x2, p = 2):
    if p == 1:
        x = torch.abs(x1 - x2)
    elif p==2:
        x = torch.sqrt(sum((x1-x2)**2))
    return x

def analyze(model):
    num_nonzeros = 0
    num_elements = 0
    count = 0
    count_0 = []

    for param_name, param in model.named_parameters():
        if "conv" in param_name:
            print(param_name)
            # num_zeros = (param.Flatten() == 0).nonzero()
            num_nonzeros += torch.count_nonzero(param)
            num_elements += param.nelement()
            count = (param.nelement() - torch.count_nonzero(param)).cpu().detach().numpy()
            count_0.append(int(count))
    
    # print(f'Number of weights reduced at each conv layer: {count_0}')
    print(f'Sparsity: {1 - num_nonzeros/num_elements}\n')
    return count_0

def get_weights(model, layer_name):
    """
    Input: 
            model: Model whose weights need to be acquired 
            layer_name: name of the layer of which the weights need to be acquired

    Output: 
            list of weights of each layer based on layer_name 
    """
    return [val.flatten().cpu().detach().numpy() for key, val in model.named_parameters() if layer_name in key and 'weight' in key]

def get_zeroIndices(wt_matrix):
    """
    Input: 
            wt_matrix: weight matrix
            
    Output: 
            list of indices where value is zero in the wt_matrix
    """
    indx = np.nonzero(wt_matrix==0)
    return indx

def get_hamming(M0, M1, name, p = 2):
    indx = {}
    wts_0 = get_weights(M0, name)
    wts_1 = get_weights(M1, name)

    # print(wts_0[0])
    if name == 'bn':
        for i in range(len(wts_0)):
            indx[i] = dist_bn(wts_0[i], wts_1[i], p)
    else:
        for i in range(len(wts_0)):
            indx[i] = PNormDist(wts_0[i], wts_1[i], p)

    return indx

def similarity(M0, M1, name):
    indx = {}
    wts_0 = get_weights(M0, name)
    wts_1 = get_weights(M1, name)
    sim = {i: [] for i in range(len(wts_0))}
    diff = {i: [] for i in range(len(wts_0))}

    for i in range(len(wts_1)):
        indx[i] = (wts_1[i]==0).nonzero()
        diff[i] = wts_0[i] - wts_1[i]
        layer = indx[i].cpu().detach().numpy().tolist()
        for ind in layer:
            if wts_0[i][ind[0]][ind[1]][ind[2]][ind[3]] != 0:
                sim[i].append(ind)

    return sim, diff

def getindices(M, layer_name, p = lambda w, j: frozenset(np.nonzero(w==j)[0].tolist())):
    """
    Input: 
            M: model
            layer_name: layer for which indices need to be fetched
            p: property based on which the indices are fetched
            
    Output: 
            dictionary of each layer containing a set of indices
    """
    zero_wts_index = {i: p(wt, 0) for i, wt in enumerate(get_weights(M, layer_name))}
    return zero_wts_index

def jaccardIndex(x, y):
    """
    Input: 
            x, y: Sets  
            
    Output: 
            Jaccard index (JI) = intersection(x, y) / union(x, y)
            
            Returns the Jaccard index for x and y
             
    """

    # intersection = len(np.intersect1d(x, y))
    # union = len(np.union1d(x, y))
    intersection_xy = len(x.intersection(y))
    union_xy = len(x.union(y))

    # print(intersection_xy, union_xy)
    # print(intersection, union)
    # Add assertion
    if union_xy == 0:
        return None
    else:
        return intersection_xy/union_xy

def plot_graphs(x, y, name):
    plt.plot(x, y)
    plt.ylabel('l2 Distance')
    plt.xlabel('BatchNorm layer')
    plt.title('L2 distance model M0 and M1')
    # plt.legend(['M0', 'M1'], loc = 'upper left')
    plt.show()
    plt.savefig(f'/u/deu9yh/compressionleakage/graphs/{name}.png')
    plt.savefig(f'/u/deu9yh/compressionleakage/graphs/{name}.pdf')

def get_num_zeros(zeros_M0, zeros_M1, flag):
    ind_conv_0 = []
    ind_conv_1 = []
    n_ind_0 = 0
    n_ind_1 = 0
    for i in zeros_M0.keys():
        if flag == 1:
            n_ind_0 += len(zeros_M0[i])
            n_ind_1 += len(zeros_M1[i])
        else:
            ind_conv_0.append(len(zeros_M0[i]))
            ind_conv_1.append(len(zeros_M1[i]))
        
    if flag == 1:
        return n_ind_0, n_ind_1
    else:
        return ind_conv_0, ind_conv_1

def calculate_JI(M0, M1, layer_name, flag):
    """
    Input: 
            M0, M1: Pytorch models
            name: Name of the layer across which the Jaccard index needs to be calculated
            flag: takes in value 1 or 2.  
            
    Output: 
            Jaccard index (JI) = intersection(x, y) / union(x, y)
            
            Returns the Jaccard index between models M0 and M1. 
            If flag = 1, returns the total number of zero indices across M0 and M1, respectively. 
            If flag = 2, returns a list of number of zero indices across layer 'name' in model M0 and M1, respectively.  
    """

    JI = []

    zero_wts_M0 = getindices(M0, layer_name)
    zero_wts_M1 = getindices(M1, layer_name)
    # print(zero_wts_M0[11])

    # print(zero_wts_M0, zero_wts_M1)

    for i in zero_wts_M0.keys():
        JI.append(jaccardIndex(zero_wts_M0[i], zero_wts_M1[i]))
    
    n_zeros_M0, n_zeros_M1 = get_num_zeros(zero_wts_M0, zero_wts_M1, flag)
    return JI, n_zeros_M0, n_zeros_M1

def get_JI(ji):
    ji_model = 0
    count = 0
    for i in ji:
        if i == None:
            pass
        else:
            ji_model += i
            count += 1
    
    ji_model = ji_model/count
    return ji_model

def get_JI_models():
    JI = []
    classname = 8
    algo = 'autocompress'
    with open(f'/u/deu9yh/compressionleakage/logs/Compressed/compression_cv/log_reports/Jaccard_index_10_models_class_{classname}_dataset_{dataset_name}_algo_{algo}.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['JI', 'Number of zeros M0', 'Number of zeros M1', 'Difference in number of zeros'])
        for i in range(10):
            M0 = copy.deepcopy(model)
            M1 = copy.deepcopy(model)
            M0.load_state_dict(torch.load(f'{directory}/models/compressed_{dataset_name}_algorithm_{algo}_M0_class_8_clip_0_sparsity_0.5_iter_{i}.pt'), strict = False)
            M1.load_state_dict(torch.load(f'{directory}/models/compressed_{dataset_name}_algorithm_{algo}_M1_class_{classname}_clip_1_sparsity_0.5_iter_{i}.pt'), strict = False)
            # M_0[i] = M0
            # M_1[i] = M1

            ji, n_ind_0, n_ind_1 = calculate_JI(M0, M1, 'conv', 1)
            # # print(ji)
            # ji_model = 0
            # count = 0
            # for i in ji:
            #     if i == None:
            #         pass
            #     else:
            #         ji_model += i
            #         count += 1
            
            # ji_model = ji_model/count

            ji_model = get_JI(ji)
            JI.append(ji_model)
            print(f'Jaccard Index: {ji_model}')
            print(f'Number of zeros in M0: {n_ind_0}')
            print(f'Number of zeros in M1: {n_ind_1}\n')
            writer.writerow([ji_model, n_ind_0, n_ind_1, abs(n_ind_0 - n_ind_1)])

    print("Mean JI: ", np.mean(JI))
    print("Standard Deviation: ", np.std(JI))

def get_JI_per_conv():
    JI = []
    n_0 = []
    n_1 = []

    with open('/u/deu9yh/compressionleakage/logs/Compressed/compression_cv/log_reports/Jaccard_index_10_models_class_6_each_conv.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Conv', 'JI', 'Number of zeros M0', 'Number of zeros M1', 'Difference in number of zeros'])
        for i in range(10):
            M0 = copy.deepcopy(model)
            M1 = copy.deepcopy(model)
            M0.load_state_dict(torch.load(f'{directory}/models/compressed_cifar10_M0_class_6_clip_0_sparsity_0.5_iter_{i}.pt'))
            M1.load_state_dict(torch.load(f'{directory}/models/compressed_cifar10_M1_class_6_clip_1_sparsity_0.5_iter_{i}.pt'))
            # M_0[i] = M0
            # M_1[i] = M1

            ji, n_ind_0, n_ind_1 = calculate_JI(M0, M1, 'conv', 2)    
            JI.append(ji)
            n_0.append(n_ind_0)   
            n_1.append(n_ind_1)
            
            # ji_model = ji_model/count

        ji_conv, num_m0, num_m1 = [], [], []

        JI = np.array(JI)
        n_0 = np.array(n_0)
        n_1 = np.array(n_1)

        for i in range(JI.shape[1]):
            JI_ = JI[:, i]
            JI_ = JI_[JI_ != None]
            ji_conv.append(np.mean(JI_))
        
        for i in range(n_0.shape[1]):
            n0_ = n_0[:, i]
            n1_ = n_1[:, i]
            n0_ = n0_[n0_ != None]
            n1_ = n1_[n1_ != None]

            num_m0.append(np.mean(n0_))
            num_m1.append(np.mean(n1_))

        print(f'Jaccard Index: {ji_conv}')
        print(f'Number of zeros in M0: {num_m0}')
        print(f'Number of zeros in M1: {num_m1}\n')

        for i, (x, y) in enumerate(zip(num_m0, num_m1)):
            writer.writerow([i, ji_conv[i], x, y, abs(x-y)])

def test_getindices(M):
    wts = get_weights(M, 'conv')
    # y = get_zeroIndices(wts[11])
    zero_wts = getindices(M, 'conv')
    x = wts[11]
    for i in zero_wts[11]:
        print(x[i])

def get_difference_wts(wts_0: np.array, wts_1: np.array):
    diff = abs(wts_1 - wts_0)
    sum = 0
    for i in range(len(diff)):
        sum += np.sum(diff[i])

    return sum

if __name__ == '__main__':
    # M0 = copy.deepcopy(model)
    # M1 = copy.deepcopy(model)
    # M0.load_state_dict(torch.load(f'{directory}/models/compressed_cifar10_M0_class_6_clip_0_sparsity_0.5_iter_0.pt'))
    # M1.load_state_dict(torch.load(f'{directory}/models/compressed_cifar10_M1_class_6_clip_1_sparsity_0.5_iter_0.pt'))
    # get_difference_wts(M0, M1)

    get_JI_models()
    # get_JI_per_conv()

    # M0 = copy.deepcopy(model)
    # M1 = copy.deepcopy(model)
    # M0.load_state_dict(torch.load(f'{directory}/models/compressed_cifar10_M1_class_6_clip_1_sparsity_0.5_iter_0_copy.pt'))
    # M1.load_state_dict(torch.load(f'{directory}/models/compressed_cifar10_M1_class_6_clip_1_sparsity_0.5_iter_1_copy.pt'))

    # # test_getindices(M1)
    # ji, n_ind_0, n_ind_1 = calculate_JI(M0, M1, 'conv', 1)
    # ji_model = 0
    # count = 0
    # for i in ji:
    #     if i == None:
    #         pass
    #     else:
    #         ji_model += i
    #         count += 1

    # ji_model = ji_model/count

    # print(f'Jaccard Index: {ji_model}')
    # print(f'Number of zeros in M0: {n_ind_0}')
    # print(f'Number of zeros in M1: {n_ind_1}\n')

    # wts_0 = get_weights(M0, 'conv')
    # wts_1 = get_weights(M1, 'conv')
    # for i in range(len(wts_0)):
    #     if all(wts_0[i].flatten() == wts_1[i].flatten()):
    #         print("same")
    #     else:
    #         print("different")

    # print(f"{'--'*10}Similarity M0 and M1{'--'*10}\n")

    # if os.path.exists('/u/deu9yh/compressionleakage/logs/Compressed/compression_cv/similarity.pickle') and os.path.exists('/u/deu9yh/compressionleakage/logs/Compressed/compression_cv/difference.pickle') and os.path.exists('/u/deu9yh/compressionleakage/logs/Compressed/compression_cv/hamming.pickle') and os.path.exists('/u/deu9yh/compressionleakage/logs/Compressed/compression_cv/l1.pickle'):
    #     with open('/u/deu9yh/compressionleakage/logs/Compressed/compression_cv/similarity.pickle', 'rb') as file:
    #         sim = pickle.load(file)
            
    #     with open('/u/deu9yh/compressionleakage/logs/Compressed/compression_cv/difference.pickle', 'rb') as file:
    #         diff = pickle.load(file)

    #     with open('/u/deu9yh/compressionleakage/logs/Compressed/compression_cv/hamming.pickle', 'rb') as file:
    #         hamming_dist_l2 = pickle.load(file)

    #     with open('/u/deu9yh/compressionleakage/logs/Compressed/compression_cv/l1.pickle', 'rb') as file:
    #         hamming_dist_l1 = pickle.load(file)

    # else:
    #     sim, diff = similarity(M0, M1, 'bn')
    #     hamming_dist_l2 = get_hamming(M0, M1, 'bn')
    #     hamming_dist_l1 = get_hamming(M0, M1, 'bn', 1)

    #     with open('/u/deu9yh/compressionleakage/logs/Compressed/compression_cv/similarity.pickle', 'wb') as file:
    #         pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)

    #     with open('/u/deu9yh/compressionleakage/logs/Compressed/compression_cv/difference.pickle', 'wb') as file:
    #         pickle.dump(diff, file, protocol=pickle.HIGHEST_PROTOCOL)

    #     with open('/u/deu9yh/compressionleakage/logs/Compressed/compression_cv/hamming.pickle', 'wb') as file:
    #         pickle.dump(hamming_dist_l2, file, protocol=pickle.HIGHEST_PROTOCOL)    

    #     with open('/u/deu9yh/compressionleakage/logs/Compressed/compression_cv/l1.pickle', 'wb') as file:
    #         pickle.dump(hamming_dist_l1, file, protocol=pickle.HIGHEST_PROTOCOL)    


    # ham = []
    # for i in hamming_dist_l2.keys():
    #     ham.append(torch.mean(hamming_dist_l2[i]).cpu().detach())

    # l1_score = []

    # for i in hamming_dist_l1.keys():
    #     l1_score.append(torch.mean(hamming_dist_l1[i]).cpu().detach())

    # plot_graphs(hamming_dist_l2.keys(), ham, 'l2_bn')


    # Add test codes for functions
