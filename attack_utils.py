import pickle as pkl
import os
from analyze_outputs import *
from simple_parsing import ArgumentParser
from pathlib import Path
from torch.utils.data import Dataset

class SubsetDataset(Dataset):
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices

    def __getitem__(self, index):
        # Fetch the image at the corresponding index
        return self.original_dataset[self.indices[index]]

    def __len__(self):
        # Return the number of images in the subset
        return len(self.indices)
    


def load_from_pickle(path):
    assert os.path.exists(path), f"Path: {path}, not found"
    with open(path, 'rb') as file:
        matrix = pickle.load(file)
    return matrix

def save_to_pickle(path, matrix):
    with open(path, 'wb') as file:
        pkl.dump(matrix, file, protocol=pkl.HIGHEST_PROTOCOL)

@torch.no_grad()
def get_loss_pretrain(M_target, M_p, d_aux):
    criterion = nn.CrossEntropyLoss()
    loss_t = 0
    loss_p = 0
    loss_diff = 0
    # progress = tqdm(enumerate(d_aux))
    for img, label, _ in d_aux:
        img = img.cuda()
        label = label.cuda()
        loss_t = criterion(M_target(img), label)
        loss_p = criterion(M_p(img), label)

        loss_diff += abs(loss_t - loss_p)

        # progress.set_description(f'Aux Data {i}/{len(d_aux)}')
    
    return loss_diff/len(d_aux) 

@torch.no_grad()
def pretrain_loss(M_p, d_aux):
    criterion = nn.CrossEntropyLoss()
    loss_p = 0
    loss_diff = 0
    # progress = tqdm(enumerate(d_aux))
    for img, label, _ in d_aux:
        img = img.cuda()
        label = label.cuda()
        loss_p = criterion(M_p(img), label)

        loss_diff += loss_p

        # progress.set_description(f'Aux Data {i}/{len(d_aux)}')
    
    return loss_diff/len(d_aux) 


@torch.no_grad()
def get_loss_per_sample(M, M_pretrain, d_aux, pretrain = False):
    loss =  [] 
    # prop = []
    criterion = nn.CrossEntropyLoss()
    # progress = tqdm(enumerate(d_aux))
    for img, label, _ in d_aux:
        img = img.cuda()
        label = label.cuda()

        loss_vic = criterion(M(img), label)
        if pretrain:
            loss_p = criterion(M_pretrain(img), label)
            loss.append(abs(loss_vic - loss_p).cpu())
            # loss.append((loss_vic - loss_p).cpu())
        else:
            loss.append(loss_vic.cpu())
        # prop.append(p)
        img.cpu()
        label.cpu()
        # p.cpu()
    
    return loss

@torch.no_grad()
def get_loss_per_data_sample(M, M_pretrain, points, pretrain = False):
    loss = []
    criterion = nn.CrossEntropyLoss()
    # progress = tqdm(enumerate(d_aux))
    for i, (img, label, _) in tqdm(enumerate(points), desc = 'Generating loss values', total = len(points)):
        # if i in points:
            # img = img.cuda()
        img = img.cuda()
        # label = label.cuda()
        label = label.cuda()
        loss_vic = criterion(M(img), label)
        if pretrain:
            loss_p = criterion(M_pretrain(img), label)
            loss.append(abs(loss_vic.item() - loss_p.item()))
            # loss.append((loss_vic - loss_p).cpu())
        else:
            loss.append(loss_vic.item())

        # property.append(p[0].item())
        img.cpu()
        label.cpu()
            
    return loss

@torch.no_grad()
def inference_per_sample(M, M_pretrain, d_aux, points, threshold, alpha, pretrain = True):
    result =  []
    criterion = nn.CrossEntropyLoss()
    M.eval()
    M_pretrain.eval()
    # progress = tqdm(enumerate(d_aux))
    # points = [i for i in range(len(threshold)) if threshold[i][0] > 1.0]
    for i, (img, label, _) in tqdm(enumerate(d_aux), total= len(d_aux), desc = f'Inferring dist'):
        if i in points:
            img = img.cuda()
            label = label.cuda()
            loss_vic = criterion(M(img), label)
            if pretrain:
                loss_p = criterion(M_pretrain(img), label)
                loss_diff = abs(loss_vic - loss_p).cpu()
                # loss_diff = (loss_vic - loss_p).cpu()
            else:
                loss_diff = abs(loss_vic).cpu()
            
            element = threshold[np.where(points == i)[0][0]]

            if element[1] == 0:
                if (loss_diff + alpha) >= element[0]:
                    result.append(0)
                else:
                    result.append(1)
            else:
                if (loss_diff + alpha) <= element[0]:
                    result.append(0)
                else:
                    result.append(1)

            # loss_0 = abs(loss_diff - threshold_0[i])
            # loss_1 = abs(loss_diff - threshold_1[i])
            
            # if loss_0 <= loss_1:
            #     result.append(0)
            # else:
            #     result.append(1)

            img.cpu()
            label.cpu()
                
        else:
            pass

    return result

@torch.no_grad()
def loss_pretrain(M_target, M_pretrain, d_aux_1, d_aux_2, pretrain= True):
    M_target.eval()
    M_pretrain.eval()
    # loss_t1, loss_t2 = 0, 0
    loss_t1, loss_t2 = [], []
    # progress = tqdm(enumerate(d_aux))
    loss_t1 = get_loss_per_sample(M_target, M_pretrain, d_aux_1, pretrain)
    # progress.set_description(f'Aux Data {i}/{len(d_aux)}')
    # mean_loss_t1 = loss_t1/len(d_aux_1)
    l2_loss_t1 = np.linalg.norm(loss_t1, 2)

    loss_t2 = get_loss_per_sample(M_target, M_pretrain, d_aux_2, pretrain)
    # mean_loss_t2 = loss_t2/len(d_aux_2)
    l2_loss_t2 = np.linalg.norm(loss_t2, 2)

    if l2_loss_t1 <= l2_loss_t2:
        return 0
    else:
        return 1

@torch.no_grad()
def get_adv_model_loss(models, M_p, d_aux, title = "", pretrain: bool = True):
    loss_models, prop_models = [], []
    for model in tqdm(models, desc = title, total = len(models)):
        model.eval()
        loss= get_loss_per_sample(model, M_p, d_aux, pretrain) 
        loss_models.append(loss)

    return loss_models

def adv_loss_thresh(M_target, M_p, threshold_0, d1, points, pretrain = True):
    M_target.cuda()
    result = inference_per_sample(M_target, M_p, d1, points, threshold_0, pretrain)
    fraction = np.mean(result)
    M_target.cpu()
    # print(result.count(0), result.count(1),"\n")
    return fraction
    # return result

@torch.no_grad()
def get_frac_thresh(models, M_p, threshold, d_aux, points, pretrain = True):
    fraction = []
    for model in models:
        model.eval()
        frac = adv_loss_thresh(model, M_p, threshold, d_aux, points, pretrain)
        fraction.append(frac)
    
    return fraction
        
def get_plot(x, y, y1: list = list(), y2: list = list(), title: str = "Graph", xlabel: str = "", ylabel: str = "", scatter: bool = False, save_path: str = "", pdf: bool = False, label_0: str = None, label_1: str = None):
    """
        If scatter = True, 
            give y1 and y2
    """
    plt.clf()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if scatter:
        assert (len(y1) != 0 or len(y2) != 0), "If scatter True, y1 and y2 are required"
        plt.plot(x, y, alpha = 0.7, color='green', marker='o', linestyle='dashed', linewidth=0.7, markersize=5, label = 'Loss difference') 
        plt.scatter(x, y1, alpha=0.5, marker= '.', c = "r", label = label_0)
        plt.scatter(x, y2, alpha=0.5, marker= '.', c = "b", label = label_1)
        plt.legend(loc = 'upper right')
    else:
        plt.plot(x, y, alpha = 0.7, color='green', marker='o', linestyle='dashed', linewidth=0.7, markersize=5)
    if pdf:
        plt.savefig(save_path)
        save_path = save_path.split(".png")[0] + ".pdf"
        plt.savefig(save_path)

@torch.no_grad()
def get_thresholds(adv_M0, adv_M1, M_p, d_aux, pretrain = True):
    loss_0, loss_1 = [], []
    M_p.eval()

    loss_0 = get_adv_model_loss(adv_M0, M_p, d_aux, title = "Generating loss values M0", pretrain = pretrain)
    mean_loss_0 = np.mean(loss_0, axis = 0)

    loss_1 = get_adv_model_loss(adv_M1, M_p, d_aux, title = "Generating loss values M1", pretrain = pretrain)
    mean_loss_1 = np.mean(loss_1, axis = 0)
    
    # points = range(len(mean_loss_0))[len(mean_loss_0)//2:]

    # ordering = np.argsort(np.abs(mean_loss_0 - mean_loss_1))
    ordering = np.argsort(mean_loss_0 - mean_loss_1)
    # index = 200
    # [-index:]
    mean_loss_0 = mean_loss_0[ordering]
    mean_loss_1 = mean_loss_1[ordering]
    threshold = [((x + y)/2, 0 if (x >= y) else 1) for x,y in zip(mean_loss_0, mean_loss_1)]

    # threshold = [(abs(x-y), 0 if (x >= y) else 1) for x,y in zip(mean_loss_0, mean_loss_1)]
    # threshold_0 = [x for x in mean_loss_0]
    # threshold_1 = [x for x in mean_loss_1]
    # print(np.std(loss_0, axis=0), np.std(loss_1, axis=0))
    return (threshold, mean_loss_0, mean_loss_1, ordering)
    # return mean_loss_0, mean_loss_8


class CustomDataset(Dataset):
    def __init__(self, loss, labels):
        self.labels = labels
        self.loss = loss
    def __len__(self):
            return len(self.labels)
    def __getitem__(self, idx):
            label = self.labels[idx]
            loss = self.loss[idx]
            # sample = {"Loss": loss, "Class": label}
            return loss, label

def data_classifier(l0, l1):
    X = torch.from_numpy(np.concatenate([l0, l1]))
    y = torch.FloatTensor(np.array([0] * len(l0) + [1] * len(l1)))
    loss_labels_df = pd.DataFrame({'Loss': X, 'Labels': y})
    
    data = CustomDataset(loss_labels_df['Loss'], loss_labels_df['Labels'])
    
    custom_transforms = transforms.Compose([
    transforms.ToTensor()
    ])
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    # dataloader = DataLoader(data, shuffle = True, batch_size= 32)

    train_set, val_set = torch.utils.data.random_split(data, [train_size, test_size])

    train_loader = DataLoader(train_set, shuffle = True, batch_size= 1)
    test_loader = DataLoader(val_set, shuffle = True, batch_size= 1) 
    
    return train_loader, test_loader

@torch.no_grad()
def compute_accuracy(model, data_loader):
    model.eval()
    correct_pred, num_examples = 0, 0
    loss = 0 
    cost = 0
    total_cost = 0
    for i, (features, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            
        features = features.to(device)
        targets = targets.to(device)
        # targets = get_smile(targets)

        logits = model(features)
        cost = F.cross_entropy(logits, targets).item()
        total_cost += cost
        # probas = F.sigmoid(model(features)) 

        _, predicted_labels = torch.max(logits, 1)
        # predicted_labels = (probas > 0.5) * 1
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        features.to("cpu")
        targets.to("cpu")
        torch.cuda.empty_cache()
    loss = total_cost/len(data_loader)
    return correct_pred.float()/num_examples * 100, loss


def train_classifier(model, train_loader, test_loader):   
    start_time = time.time()
    learning_rate = 1e-3
    num_epochs = 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        
        model.train()
        model = model.to(device)
        progress = tqdm(enumerate(train_loader))
        loss_so_far = 0
        for batch_idx, (features, targets) in progress:
            
            optimizer.zero_grad()
            features = features.to(device)
            targets = targets.to(device)
                
            ### FORWARD AND BACK PROP
            logits= model(features)
            # probas = F.softmax(logits, dim = 1)
            cost = F.cross_entropy(logits, targets)
            # optimizer.zero_grad()

            cost.backward()
            
            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            with torch.no_grad():
                loss_so_far += cost.item()
            
            ### LOGGING
            if not batch_idx % 50:
                progress.set_description('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f' 
                    %(epoch+1, num_epochs, batch_idx, 
                        len(train_loader), loss_so_far / (batch_idx + 1)))
            features.to("cpu")
            targets.to("cpu")
            torch.cuda.empty_cache()

        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            train_acc, train_loss = compute_accuracy(model, train_loader)
            val_acc, val_loss = compute_accuracy(model, test_loader)
            print('Epoch: %03d/%03d | Accuracy Train: %.3f%% | Loss Train: %.3f | Accuracy Valid: %.3f%% | Loss Valid: %.3f' % (
                epoch+1, num_epochs, 
                train_acc, train_loss,
                val_acc, val_loss
                ))
            
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    torch.save(model.state_dict(), "/p/compressionleakage/logs/Compressed/compression_cv/models/celeba/classifier_celeba_smiling.pt")
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    return model
