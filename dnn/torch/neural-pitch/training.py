"""
Training the neural pitch estimator

"""

import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('features', type=str, help='.f32 IF Features for training (generated by augmentation script)')
parser.add_argument('features_pitch', type=str, help='.npy Pitch file for training (generated by augmentation script)')
parser.add_argument('output_folder', type=str, help='Output directory to store the model weights and config')
parser.add_argument('data_format', type=str, help='Choice of Input Data',choices=['if','xcorr','both'])
parser.add_argument('--gpu_index', type=int, help='GPU index to use if multiple GPUs',default = 0,required = False)
parser.add_argument('--confidence_threshold', type=float, help='Confidence value below which pitch will be neglected during training',default = 0.4,required = False)
parser.add_argument('--context', type=int, help='Sequence length during training',default = 100,required = False)
parser.add_argument('--N', type=int, help='STFT window size',default = 320,required = False)
parser.add_argument('--H', type=int, help='STFT Hop size',default = 160,required = False)
parser.add_argument('--xcorr_dimension', type=int, help='Dimension of Input cross-correlation',default = 257,required = False)
parser.add_argument('--freq_keep', type=int, help='Number of Frequencies to keep',default = 30,required = False)
parser.add_argument('--gru_dim', type=int, help='GRU Dimension',default = 64,required = False)
parser.add_argument('--output_dim', type=int, help='Output dimension',default = 192,required = False)
parser.add_argument('--learning_rate', type=float, help='Learning Rate',default = 1.0e-3,required = False)
parser.add_argument('--epochs', type=int, help='Number of training epochs',default = 50,required = False)
parser.add_argument('--choice_cel', type=str, help='Choice of Cross Entropy Loss (default or robust)',choices=['default','robust'],default = 'default',required = False)
parser.add_argument('--prefix', type=str, help="prefix for model export, default: model", default='model')


args = parser.parse_args()

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

# Fixing the seeds for reproducability
import time
np_seed = int(time.time())
torch_seed = int(time.time())

import torch
torch.manual_seed(torch_seed)
import numpy as np
np.random.seed(np_seed)
from utils import count_parameters
import tqdm
from models import PitchDNN, PitchDNNIF, PitchDNNXcorr, PitchDNNDataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if args.data_format == 'if':
    pitch_nn = PitchDNNIF(3 * args.freq_keep - 2, args.gru_dim, args.output_dim)
elif args.data_format == 'xcorr':
    pitch_nn = PitchDNNXcorr(args.xcorr_dimension, args.gru_dim, args.output_dim)
else:
    pitch_nn = PitchDNN(3 * args.freq_keep - 2, 224, args.gru_dim, args.output_dim)

dataset_training = PitchDNNDataloader(args.features,args.features_pitch,args.confidence_threshold,args.context,args.data_format)

def loss_custom(logits,labels,confidence,choice = 'default',nmax = 192,q = 0.7):
    logits_softmax = torch.nn.Softmax(dim = 1)(logits).permute(0,2,1)
    labels_one_hot = torch.nn.functional.one_hot(labels.long(),nmax)

    if choice == 'default':
        # Categorical Cross Entropy
        CE = -torch.sum(torch.log(logits_softmax*labels_one_hot + 1.0e-6)*labels_one_hot,dim=-1)
        CE = torch.mean(confidence*CE)

    else:
        # Robust Cross Entropy
        CE = (1.0/q)*(1 - torch.sum(torch.pow(logits_softmax*labels_one_hot + 1.0e-7,q),dim=-1) )
        CE = torch.sum(confidence*CE)

    return CE

def accuracy(logits,labels,confidence,choice = 'default',nmax = 192,q = 0.7):
    logits_softmax = torch.nn.Softmax(dim = 1)(logits).permute(0,2,1)
    pred_pitch = torch.argmax(logits_softmax, 2)
    accuracy = (pred_pitch != labels.long())*1.
    return 1.-torch.mean(confidence*accuracy)

train_dataset, test_dataset = torch.utils.data.random_split(dataset_training, [0.95,0.05], generator=torch.Generator().manual_seed(torch_seed))

batch_size = 256
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

pitch_nn = pitch_nn.to(device)
num_params = count_parameters(pitch_nn)
learning_rate = args.learning_rate
model_opt = torch.optim.Adam(pitch_nn.parameters(), lr = learning_rate)

num_epochs = args.epochs

for epoch in range(num_epochs):
    losses = []
    accs = []
    pitch_nn.train()
    with tqdm.tqdm(train_dataloader) as train_epoch:
        for i, (xi, yi, ci) in enumerate(train_epoch):
            yi, xi, ci = yi.to(device, non_blocking=True), xi.to(device, non_blocking=True), ci.to(device, non_blocking=True)
            pi = pitch_nn(xi.float())
            loss = loss_custom(logits = pi,labels = yi,confidence = ci,choice = args.choice_cel,nmax = args.output_dim)
            acc = accuracy(logits = pi,labels = yi,confidence = ci,choice = args.choice_cel,nmax = args.output_dim)
            acc = acc.detach()

            model_opt.zero_grad()
            loss.backward()
            model_opt.step()

            losses.append(loss.item())
            accs.append(acc.item())
            avg_loss = np.mean(losses)
            avg_acc = np.mean(accs)
            train_epoch.set_postfix({"Train Epoch" : epoch, "Train Loss":avg_loss, "acc" : avg_acc.item()})

    if epoch % 5 == 0:
        pitch_nn.eval()
        losses = []
        with tqdm.tqdm(test_dataloader) as test_epoch:
            for i, (xi, yi, ci) in enumerate(test_epoch):
                yi, xi, ci = yi.to(device, non_blocking=True), xi.to(device, non_blocking=True), ci.to(device, non_blocking=True)
                pi = pitch_nn(xi.float())
                loss = loss_custom(logits = pi,labels = yi,confidence = ci,choice = args.choice_cel,nmax = args.output_dim)
                losses.append(loss.item())
                avg_loss = np.mean(losses)
                test_epoch.set_postfix({"Epoch" : epoch, "Test Loss":avg_loss})

pitch_nn.eval()

config = dict(
    data_format=args.data_format,
    epochs=num_epochs,
    window_size= args.N,
    hop_factor= args.H,
    freq_keep=args.freq_keep,
    batch_size=batch_size,
    learning_rate=learning_rate,
    confidence_threshold=args.confidence_threshold,
    model_parameters=num_params,
    np_seed=np_seed,
    torch_seed=torch_seed,
    xcorr_dim=args.xcorr_dimension,
    dim_input=3*args.freq_keep - 2,
    gru_dim=args.gru_dim,
    output_dim=args.output_dim,
    choice_cel=args.choice_cel,
    context=args.context,
)

model_save_path = os.path.join(args.output_folder, f"{args.prefix}_{args.data_format}.pth")
checkpoint = {
    'state_dict': pitch_nn.state_dict(),
    'config': config
}
torch.save(checkpoint, model_save_path)
