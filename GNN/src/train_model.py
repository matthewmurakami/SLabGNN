import os
import torch
import torch_geometric
import numpy as np
import pickle
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.nn import MSELoss
unicode = str
from datetime import datetime
from scipy import ndimage, stats
from torch.nn import Linear, ReLU,CrossEntropyLoss
from tqdm import tqdm

def train(train_loader, val_loader, model, output_filepath, learning_rate, early_stopping_count, early_stopping_loss_eps):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)



    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1000):
        model.train()
        train_loss = 0.0

        for data in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data)
                print("DEBUG1 ======================> ", output)
                loss = criterion(output, data.y)
                print("DEBUG2 ======================> ", data.y.shape)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss - early_stopping_loss_eps:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), output_filepath)
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_count:
            print('Early stopping triggered.')
            break

    print('Training finished.')
    model.load_state_dict(torch.load(output_filepath))
    model.eval()

    # Evaluate the model on the validation set
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            output = model(data)
            print(output.shape, data.y.shape)
            loss = criterion(output, data.y)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Final Validation Loss: {val_loss:.4f}')











































# def write_header(output_filename):
    # fieldnames = ['Epoch', 'Time stamp', 'ID', 'Name', 'Serial', 'UUID', 'GPU temp. [C]', 'GPU util. [%]', 'Memory util. [%]',
    #               'Memory total [MB]', 'Memory used [MB]', 'Memory free [MB]', 'Display mode', 'Display active']
    # with open(output_filename, 'w', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()

# def record(epoch, output_filename):
#     #out_name, out_file_extension = os.path.splitext(output_filename)
#     output_dir = os.path.dirname(output_filename)

#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)

#     GPUs = GPUtil.getGPUs()
#     print('INFO: GPUs:', GPUs)
#     if len(GPUs) < 1:
#         print('WARNING: the hardware does not contain NVIDIA GPU card')
#         return

#     now = datetime.now()  # current date and time
#     date_time = now.strftime("%Y:%m:%d:%H:%M:%S")
#     print('INFO: date_time ', date_time)

#     attrList = [[{'attr': 'id', 'name': 'ID'},
#                  {'attr': 'name', 'name': 'Name'},
#                  {'attr': 'serial', 'name': 'Serial'},
#                  {'attr': 'uuid', 'name': 'UUID'}],
#                 [{'attr': 'temperature', 'name': 'GPU temp.', 'suffix': 'C', 'transform': lambda x: x, 'precision': 0},
#                  {'attr': 'load', 'name': 'GPU util.', 'suffix': '%', 'transform': lambda x: x * 100, 'precision': 0},
#                  {'attr': 'memoryUtil', 'name': 'Memory util.', 'suffix': '%', 'transform': lambda x: x * 100,
#                   'precision': 0}],
#                 [{'attr': 'memoryTotal', 'name': 'Memory total', 'suffix': 'MB', 'precision': 0},
#                  {'attr': 'memoryUsed', 'name': 'Memory used', 'suffix': 'MB', 'precision': 0},
#                  {'attr': 'memoryFree', 'name': 'Memory free', 'suffix': 'MB', 'precision': 0}],
#                 [{'attr': 'display_mode', 'name': 'Display mode'},
#                  {'attr': 'display_active', 'name': 'Display active'}]]


#     # store the date_time as teh first entry in the recorded row
#     store_gpu_info = str(epoch) + ',' + date_time

#     for attrGroup in attrList:
#         #print('INFO: attrGroup:', attrGroup)

#         index = 1
#         for attrDict in attrGroup:
#             attrPrecision = '.' + str(attrDict['precision']) if ('precision' in attrDict.keys()) else ''
#             attrTransform = attrDict['transform'] if ('transform' in attrDict.keys()) else lambda x: x

#             for gpu in GPUs:
#                 attr = getattr(gpu, attrDict['attr'])

#                 attr = attrTransform(attr)

#                 if (isinstance(attr, float)):
#                     attrStr = ('{0:' + attrPrecision + 'f}').format(attr)
#                 elif (isinstance(attr, int)):
#                     attrStr = ('{0:d}').format(attr)
#                 elif (isinstance(attr, str)):
#                     attrStr = attr;
#                 elif (sys.version_info[0] == 2):
#                     if (isinstance(attr, unicode)):
#                         attrStr = attr.encode('ascii', 'ignore')
#                 else:
#                     raise TypeError(
#                         'Unhandled object type (' + str(type(attr)) + ') for attribute \'' + attrDict['name'] + '\'')

#                 #print('INFO: attrStr ', attrStr)
#                 store_gpu_info += ',' + attrStr
#                 index +=1

#     store_gpu_info += '\n'
#     print('row data:', store_gpu_info)
#     with open(output_filename, 'a', newline='') as csvfile:
#         csvfile.write(store_gpu_info)

# def compute_metrics(x, y, name, epoch):
#     # convert x to numpy
#     x = x.detach().cpu().numpy()
#     y = y.detach().cpu().numpy()

#     # convert x from one hot to class label
#     x = np.argmax(x, axis=1)  # assumes NCHW tensor order

#     assert x.shape == y.shape

#     # flatten into a 1d vector
#     x = x.flatten()
#     y = y.flatten()

#     pass

# def eval_model(model, dataloader, criterion, device, epoch, name):
#     print("eval")
#     avg_loss = 0
#     batch_count = len(dataloader)
#     model.eval()

#     with torch.no_grad():
#         for batch in dataloader:
#             data_input = batch.to(device)
#             pred = model(data_input)
            
#             # compute metrics
#             masks = data_input.y.to(torch.float32)   # was torch.int64
#             batch_train_loss = criterion(pred, masks)
#             avg_loss += batch_train_loss.item()

#     avg_loss /= batch_count

#     return avg_loss

# def train_epoch(model, dataloader, optimizer, criterion, lr_scheduler, device, epoch):
#     avg_train_loss = 0
#     model.train()
#     batch_count = len(dataloader)
    
#     for batch in dataloader:
#         optimizer.zero_grad()
#         data_input = batch.to(device)
#         print(data_input)
#         pred = model(data_input)

#         # compute metrics
#         masks = data_input.y.to(torch.float32)  # was torch.int64
#         batch_train_loss = criterion(pred, masks)
            
#         batch_train_loss.backward()
#         avg_train_loss += batch_train_loss.item()

#         optimizer.step()

#         if lr_scheduler is not None:
#             lr_scheduler.step()

#     avg_train_loss /= batch_count

#     return model

# def train(train_dataset, val_dataset, model, output_filepath, learning_rate, early_stopping_epoch_count=5, loss_eps=1e-3):
#     print("training")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print (device)
    
#     if not os.path.exists(output_filepath):
#         os.makedirs(output_filepath)

#     train_dl = DataLoader(train_dataset, batch_size=1, shuffle = True)
#     val_dl = DataLoader(val_dataset, batch_size=1, shuffle = True)

#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = MSELoss()
#     model = model.to(device)
#     epoch = 0
#     done = False
#     best_model = model

#     best_val_loss = float('inf')
#     best_val_loss_epoch = 0

#     while epoch<1000:
#         print('Epoch: {}'.format(epoch))

#         model = train_epoch(model, train_dl, optimizer, criterion, None, device, epoch)
#         val_loss = eval_model(model, val_dl, criterion, device, epoch, 'val')
#         error_from_best = np.abs(val_loss - np.min(val_loss))
#         error_from_best[error_from_best < np.abs(loss_eps)] = 0

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model = copy.deepcopy(model)
#             best_val_loss_epoch = epoch

#         if epoch >= (best_val_loss_epoch + early_stopping_epoch_count):
#             print("Exiting training loop in epoch: {} - due to early stopping criterion being met".format(epoch))
#             break
#         epoch += 1

#     if val_dl is not None:
#         print('Evaluating model against test dataset')
#         eval_model(model, val_dl, criterion, device, epoch, 'test')

#     best_model.cpu()
#     torch.save(best_model.state_dict(), os.path.join(output_filepath, 'model.pt'))



"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train_model(train_dataset, val_dataset, model, output_filepath, learning_rate, early_stopping_epoch_count=5, loss_eps=1e-3):
    train_dl = DataLoader(train_dataset, batch_size=1, shuffle = True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle = True)
    
    for epoch in range(1, 1000):
        train(model, train_dl, output_filepath, learning_rate)
        train_acc = test(train_dl)
        test_acc = test(model, val_dl)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
def train(model, train_dl, output_filepath, learning_rate):
    model.train()
    criterion = MSELoss()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for data in train_dl:  # Iterate in batches over the training dataset.
         out = model(data, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(model, val_dl):
     model.eval()

     correct = 0
     for data in val_dl:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(val_dl.dataset)  # Derive ratio of correct predictions.
"""