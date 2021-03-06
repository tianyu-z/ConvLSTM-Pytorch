from comet_ml import Experiment, ExistingExperiment
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import get_params
from data.cloudcast import CloudCast
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from utils import flatten_opts, psnr, ssim, upload_images

# from tensorboardX import SummaryWriter
import argparse

TIMESTAMP = "2021-03-25T00-00-00"
parser = argparse.ArgumentParser()
parser.add_argument(
    "-clstm", "--convlstm", help="use convlstm as base cell", action="store_true"
)
parser.add_argument(
    "-cgru", "--convgru", help="use convgru as base cell", action="store_true"
)
parser.add_argument("-bs", "--batchsize", default=16, type=int, help="mini-batch size")
parser.add_argument(
    "-nw", "--num_workers", default=4, type=int, help="number of CPU you get"
)
parser.add_argument(
    "-dh", "--data_h", default=128, type=int, help="H of the data shape"
)
parser.add_argument(
    "-dw", "--data_w", default=128, type=int, help="W of the data shape"
)
parser.add_argument(
    "-se", "--save_every", default=5, type=int, help="save for every x epoches"
)
parser.add_argument(
    "-ct", "--continue_train", action="store_true", help="resume an epoch"
)
parser.add_argument(
    "--checkpoint", type=str, help="path of checkpoint for pretrained model"
)
parser.add_argument("-lr", default=1e-4, type=float, help="G learning rate")
parser.add_argument("-frames_input", default=10, type=int, help="sum of input frames")
parser.add_argument(
    "-frames_output", default=10, type=int, help="sum of predict frames"
)
parser.add_argument(
    "--cometid", type=str, default="", help="the comet id to resume exps",
)
parser.add_argument("-epochs", default=500, type=int, help="sum of epochs")
parser.add_argument(
    "-wp", "--workspace", default="tianyu-z", type=str, help="comet-ml workspace"
)
parser.add_argument(
    "-pn",
    "--projectname",
    default="ccml_convlstm",
    type=str,
    help="comet-ml project name",
)
parser.add_argument(
    "--nocomet", action="store_true", help="not using comet_ml logging."
)
parser.add_argument(
    "-cd", "--checkdata", action="store_true", help="not using comet_ml logging."
)
args = parser.parse_args()

# start logging info in comet-ml
if not args.nocomet:
    comet_exp = Experiment(workspace=args.workspace, project_name=args.projectname)
    # comet_exp.log_parameters(flatten_opts(args))
else:
    comet_exp = None
if not args.nocomet and args.cometid != "":
    comet_exp = ExistingExperiment(previous_experiment=args.cometid)
elif not args.nocomet and args.cometid == "":
    comet_exp = Experiment(workspace=args.workspace, project_name=args.projectname)
else:
    comet_exp = None
random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

save_dir = "./save_model/" + TIMESTAMP

trainFolder = CloudCast(
    is_train=True,
    root="data/",
    n_frames_input=args.frames_input,
    n_frames_output=args.frames_output,
    batchsize=args.batchsize,
)
validFolder = CloudCast(
    is_train=False,
    root="data/",
    n_frames_input=args.frames_input,
    n_frames_output=args.frames_output,
    batchsize=args.batchsize,
)
trainLoader = torch.utils.data.DataLoader(
    trainFolder, batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False
)
validLoader = torch.utils.data.DataLoader(
    validFolder, batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False
)
(
    convlstm_encoder_params,
    convlstm_decoder_params,
    convgru_encoder_params,
    convgru_decoder_params,
) = get_params([args.data_h, args.data_w])
if args.convlstm:
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
if args.convgru:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
else:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params


def train(exp=None):
    """
    main function to run the training
    """
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder)
    run_dir = "./runs/" + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    # tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(args.checkpoint) and args.continue_train:
        # load existing model
        print("==> loading existing model")
        model_info = torch.load(args.checkpoint)
        net.load_state_dict(model_info["state_dict"])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info["optimizer"])
        cur_epoch = model_info["epoch"] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=4, verbose=True
    )

    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # pnsr ssim
    avg_psnrs = {}
    avg_ssims = {}
    for j in range(args.frames_output):
        avg_psnrs[j] = []
        avg_ssims[j] = []
    if args.checkdata:
        # Checking dataloader
        print("Checking Dataloader!")
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            assert targetVar.shape == torch.Size(
                [args.batchsize, args.frames_output, 1, args.data_h, args.data_w]
            )
            assert inputVar.shape == torch.Size(
                [args.batchsize, args.frames_input, 1, args.data_h, args.data_w]
            )
        print("TrainLoader checking is complete!")
        t = tqdm(validLoader, leave=False, total=len(validLoader))
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            assert targetVar.shape == torch.Size(
                [args.batchsize, args.frames_output, 1, args.data_h, args.data_w]
            )
            assert inputVar.shape == torch.Size(
                [args.batchsize, args.frames_input, 1, args.data_h, args.data_w]
            )
        print("ValidLoader checking is complete!")
        # mini_val_loss = np.inf
    for epoch in range(cur_epoch, args.epochs + 1):
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        psnr_dict = {}
        ssim_dict = {}
        for j in range(args.frames_output):
            psnr_dict[j] = 0
            ssim_dict[j] = 0
        image_log = []
        if exp is not None:
            exp.log_metric("epoch", epoch)
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            inputs = inputVar.to(device)  # B,S,C,H,W
            label = targetVar.to(device)  # B,S,C,H,W
            optimizer.zero_grad()
            net.train()
            pred = net(inputs)  # B,S,C,H,W
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / args.batchsize
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            t.set_postfix(
                {
                    "trainloss": "{:.6f}".format(loss_aver),
                    "epoch": "{:02d}".format(epoch),
                }
            )
        # tb.add_scalar('TrainLoss', loss_aver, epoch)
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                pred = net(inputs)
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / args.batchsize
                # record validation loss
                valid_losses.append(loss_aver)

                for j in range(args.frames_output):
                    psnr_dict[j] += psnr(pred[:, j], label[:, j])
                    ssim_dict[j] += ssim(pred[:, j], label[:, j])
                # print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix(
                    {
                        "validloss": "{:.6f}".format(loss_aver),
                        "epoch": "{:02d}".format(epoch),
                    }
                )
                if i % 500 == 499:
                    for k in range(args.frames_output):
                        image_log.append(label[0, k].unsqueeze(0).repeat(1, 3, 1, 1))
                        image_log.append(pred[0, k].unsqueeze(0).repeat(1, 3, 1, 1))
                    upload_images(
                        image_log,
                        epoch,
                        exp=exp,
                        im_per_row=2,
                        rows_per_log=int(len(image_log) / 2),
                    )
        # tb.add_scalar('ValidLoss', loss_aver, epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        for j in range(args.frames_output):
            avg_psnrs[j].append(psnr_dict[j] / i)
            avg_ssims[j].append(ssim_dict[j] / i)
        epoch_len = len(str(args.epochs))

        print_msg = (
            f"[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] "
            + f"train_loss: {train_loss:.6f} "
            + f"valid_loss: {valid_loss:.6f}"
            + f"PSNR_1: {psnr_dict[0] / i:.6f}"
            + f"SSIM_1: {ssim_dict[0] / i:.6f}"
        )

        # print(print_msg)
        # clear lists to track next epoch
        if exp is not None:
            exp.log_metric("TrainLoss", train_loss)
            exp.log_metric("ValidLoss", valid_loss)
            exp.log_metric("PSNR_1", psnr_dict[0] / i)
            exp.log_metric("SSIM_1", ssim_dict[0] / i)
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            "epoch": epoch,
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "avg_psnrs": avg_psnrs,
            "avg_ssims": avg_ssims,
            "avg_valid_losses": avg_valid_losses,
            "avg_train_losses": avg_train_losses,
        }
        save_flag = False
        if epoch % args.save_every == 0:
            torch.save(
                model_dict,
                save_dir
                + "/"
                + "checkpoint_{}_{:.6f}.pth".format(epoch, valid_loss.item()),
            )
            print("Saved" + "checkpoint_{}_{:.6f}.pth".format(epoch, valid_loss.item()))
            save_flag = True
        if avg_psnrs[0][-1] == max(avg_psnrs[0]) and not save_flag:
            torch.save(
                model_dict, save_dir + "/" + "bestpsnr_1.pth",
            )
            print("Best psnr found and saved")
            save_flag = True
        if avg_ssims[0][-1] == max(avg_ssims[0]) and not save_flag:
            torch.save(
                model_dict, save_dir + "/" + "bestssim_1.pth",
            )
            print("Best ssim found and saved")
            save_flag = True
        if avg_valid_losses[-1] == min(avg_valid_losses) and not save_flag:
            torch.save(
                model_dict, save_dir + "/" + "bestvalidloss.pth",
            )
            print("Best validloss found and saved")
            save_flag = True
        if not save_flag:
            torch.save(
                model_dict, save_dir + "/" + "checkpoint.pth",
            )
            print("The latest normal checkpoint saved")
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", "wt") as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", "wt") as f:
        for i in avg_valid_losses:
            print(i, file=f)


if __name__ == "__main__":
    train(exp=comet_exp)
