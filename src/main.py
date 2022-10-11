import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from model import Encoder, Decoder, ConvLSTM
from dataset import MovingMNIST
from datetime import date, datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def predictor(previous_seq, previous_num, prediction_num, encoder, decoder, convlstms):
    # at first 'previous_seq' has size (batch, height, width, seq)
    _, frame_height, frame_width, _ = previous_seq.size()
    encoder_input = previous_seq.permute(0, 3, 1, 2).contiguous() # make shape (batch, seq, height, width)
    encoder_input = encoder_input.view([-1, frame_height, frame_width]) # make shape (batch * seq, height, width)
    encoder_input = encoder_input.unsqueeze(1) # make shape (batch * seq, channel=1, height, width)

    # Encoder with convolution layers
    enc_feature = encoder(encoder_input)
    _, feature_channel, feature_height, feature_width = enc_feature.size()

    # make shape (batch, seq, channel, height, width) to separate batch and seq
    enc_feature = enc_feature.view([-1, previous_num, feature_channel, feature_height, feature_width])

    # ConvLSTM

    conv_lstm_h1 = torch.zeros_like(enc_feature[:, 0]) # batch, channel, height, width
    conv_lstm_c1 = torch.zeros_like(enc_feature[:, 0])
    conv_lstm_h2 = torch.zeros_like(enc_feature[:, 0])
    conv_lstm_c2 = torch.zeros_like(enc_feature[:, 0])
    unit1, unit2 = convlstms # lstm blocks

    for seq_i in range(previous_num + prediction_num - 3):
        conv_lstm_h1, conv_lstm_c1 = unit1(enc_feature[:, seq_i], h=conv_lstm_h1, c=conv_lstm_c1)
        conv_lstm_h2, conv_lstm_c2 = unit2(conv_lstm_h1, h=conv_lstm_h2, c=conv_lstm_c2)

    #print('LSTM output shape: {}'.format(conv_lstm_h2.shape)) # (128, 64, 16, 16)

    decoder_input_1 = conv_lstm_h2.unsqueeze(1) # (128, 1, 64, 16, 16)

    conv_lstm_h1, conv_lstm_c1 = unit1(torch.zeros_like(enc_feature[:, 0]), h=conv_lstm_h1, c=conv_lstm_c1)
    conv_lstm_h2, conv_lstm_c2 = unit2(conv_lstm_h1, h=conv_lstm_h2, c=conv_lstm_c2)

    decoder_input_2 = conv_lstm_h2.unsqueeze(1)

    conv_lstm_h1, conv_lstm_c1 = unit1(torch.zeros_like(enc_feature[:, 0]), h=conv_lstm_h1, c=conv_lstm_c1)
    conv_lstm_h2, conv_lstm_c2 = unit2(conv_lstm_h1, h=conv_lstm_h2, c=conv_lstm_c2)

    decoder_input_3 = conv_lstm_h2.unsqueeze(1)

    decoder_input = torch.cat((decoder_input_1, decoder_input_2, decoder_input_3), dim=1) # (128, 3, 64, 16, 16)
    decoder_input = decoder_input.reshape(-1, decoder_input.shape[2], decoder_input.shape[3], decoder_input.shape[4]) # (384, 64, 16, 16)

    # Decoder with deconvolution layers
    prediction = decoder(decoder_input)

    prediction = prediction.squeeze(1) # (batch*seq, _ )
    prediction = prediction.reshape(-1, 3, prediction.shape[1], prediction.shape[2]) # (batch, seq, _ )

    # final 'prediction' should have (batch, height, width, 3) to calcuate the loss
    prediction = prediction.permute(0, 2, 3, 1).contiguous()
    return prediction

def main():
# define the optimizer parameters, and learning parameters
    previous_num = 5
    prediction_num = 3
    minibatch_size = 128
    lear_rate = 0.005
    bta1 = 0.9
    bta2 = 0.999
    epsln = 1e-8
    day = date.today().day
    current_time = datetime.now().strftime("%H_%M_%S")
    LOGDIR = './MovingMNIST_video_tutorial2/'
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    writer = SummaryWriter(log_dir=LOGDIR + 'Test_' + '%d' % day + '_' + current_time)

    ##################### data
    train_dataset = MovingMNIST(fdir='./mnist_test_seq.npy', split='train', previous_num=previous_num, prediction_num=prediction_num)
    test_dataset = MovingMNIST(fdir='./mnist_test_seq.npy', split='test', previous_num=previous_num, prediction_num=prediction_num)

    # Create dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=minibatch_size, shuffle=False)

    ##################### model
    encoder = Encoder().cuda()
    decoder = Decoder().cuda()
    convlstm1 = ConvLSTM(in_channels=64, kernel_size=5, filters=64, padding=2).cuda()
    convlstm2 = ConvLSTM(in_channels=64, kernel_size=5, filters=64, padding=2).cuda()
    convlstms = [convlstm1, convlstm2]
    criterion = nn.MSELoss(reduction='sum')
    params = [{'params': encoder.parameters()},
                {'params': decoder.parameters()},
                {'params': convlstm1.parameters()},
                {'params': convlstm2.parameters()},
    ]
    optimizer = torch.optim.Adam(params, lr=lear_rate, betas=(bta1, bta2), eps=epsln)
    step = 0
    while step < 20001:
        for _, (previous_seq, gt_next_frame) in enumerate(train_loader):
            encoder.train(), decoder.train(), convlstm1.train(), convlstm2.train()
            step += 1
            previous_seq = previous_seq.cuda()
            gt_next_frame = gt_next_frame.cuda()
            predicted_next_frame = predictor(previous_seq, previous_num, prediction_num, encoder, decoder, convlstms)
            pixel_loss = criterion(predicted_next_frame, gt_next_frame) / (prediction_num * minibatch_size)
            optimizer.zero_grad()
            pixel_loss.backward()
            optimizer.step()
            if (step == 3000):
                optimizer = torch.optim.SGD(params, lr=lear_rate*10)
            if (step) % 100 == 0: # test at every 100 steps
                with torch.no_grad():
                    encoder.eval(), decoder.eval(), convlstm1.eval(), convlstm2.eval()
                    for _, (test_previous_seq, test_gt_next_frame) in enumerate(test_loader):
                        test_previous_seq = test_previous_seq.cuda()
                        test_gt_next_frame = test_gt_next_frame.cuda()
                        test_predicted_next_frame = predictor(test_previous_seq, previous_num, prediction_num, encoder, decoder, convlstms)
                        test_loss = criterion(test_predicted_next_frame, test_gt_next_frame) / (prediction_num * test_previous_seq.size(0))
                print('@ iteration: %i, Training loss = %.6f, Test loss = %.6f' % (step, pixel_loss.cpu().item(), test_loss.cpu().item()))

                # put images on the tensorboard
                writer.add_images('predicted_next_frame/0', (test_predicted_next_frame[0:1].permute(0, 3, 1, 2).detach().cpu().numpy() + 1) / 2, step)
                writer.add_images('predicted_next_frame/1', (test_predicted_next_frame[1:2].permute(0, 3, 1, 2).detach().cpu().numpy() + 1) / 2, step)
                writer.add_images('predicted_next_frame/2', (test_predicted_next_frame[2:3].permute(0, 3, 1, 2).detach().cpu().numpy() + 1) / 2, step)

                # add_images: image shape should be T 1 H W, and have data range 0 ~ 1
                writer.add_images('gt_next_frame/0', (test_gt_next_frame[0:1].permute(0, 3, 1, 2).cpu().numpy() + 1) / 2)
                writer.add_images('gt_next_frame/1', (test_gt_next_frame[1:2].permute(0, 3, 1, 2).cpu().numpy() + 1) / 2)
                writer.add_images('gt_next_frame/2', (test_gt_next_frame[2:3].permute(0, 3, 1, 2).cpu().numpy() + 1) / 2)
                writer.add_images('input_frames/0', (test_previous_seq[0:1, :, :, :previous_num].permute(3, 0, 1, 2).cpu().numpy() + 1) / 2)
                writer.add_images('input_frames/1', (test_previous_seq[1:2, :, :, :previous_num].permute(3, 0, 1, 2).cpu().numpy() + 1) / 2)
                writer.add_images('input_frames/2', (test_previous_seq[2:3, :, :, :previous_num].permute(3, 0, 1, 2).cpu().numpy() + 1) / 2)

                # save the traind model
                encoder_state_dict = encoder.state_dict()
                decoder_state_dict = decoder.state_dict()
                convlstm1_state_dict = convlstm1.state_dict()
                convlstm2_state_dict = convlstm2.state_dict()
                torch.save({'encoder_state_dict': encoder_state_dict,
                        'decoder_state_dict': decoder_state_dict,
                        'convlstm1_state_dict': convlstm1_state_dict,
                        'convlstm2_state_dict': convlstm2_state_dict},
                        os.path.join(LOGDIR, 'model_step%i.ckpt' % (step)))
if __name__ =='__main__':
    main()
