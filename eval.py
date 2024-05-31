import time
from datetime import datetime

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from datasets import *
from utils import *
import torch.nn.functional as F
from tqdm import tqdm

from utils_eval import get_bleu, get_rouge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for thesis and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to thesis are fixed size; otherwise lots of computational overhead

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def eval_validate(val_loader, encoder, decoder, word_map, write_log=False, eval_len=0, save_img=True,
                  flag=""):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder thesis
    :param decoder: decoder thesis
    :param criterion: loss layer
    :param word_map: Word map.
    :return: BLEU-4 score
    """
    global writer
    # TODO
    decoder.eval()  # eval mode (no dropout or batchnorm)
    encoder.eval()

    bleu = AverageMeter()
    bleu1 = AverageMeter()
    bleu2 = AverageMeter()
    bleu3 = AverageMeter()
    bleu4 = AverageMeter()
    rouge = AverageMeter()
    rouge1 = AverageMeter()
    rouge2 = AverageMeter()
    rouge3 = AverageMeter()
    rouge4 = AverageMeter()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        # Batches
        with tqdm(total=len(val_loader), desc=f"Eval-Processing") as t:
            for i, (imgs, caps, caplens, allcaps, pri_image) in enumerate(val_loader):
                # Move to device, if available
                imgs = imgs.to(device)
                caps = caps.to(device)
                caplens = caplens.to(device)
                allcaps = allcaps.to(device)

                # Forward prop.
                if encoder is not None:
                    imgs = encoder(imgs)
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

                scores_copy = scores.clone()

                # References
                allcaps = allcaps[sort_ind]  # because images were sorted in the decoder

                for j in range(allcaps.shape[0]):
                    img_caps = allcaps[j].tolist()
                    img_captions = list(
                        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                            img_caps))  # remove <start> and pads
                    references.append(img_captions)

                # Hypotheses
                _, preds = torch.max(scores_copy, dim=2)
                preds = preds.tolist()
                temp_preds = list()
                for j, p in enumerate(preds):
                    temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
                preds = temp_preds
                hypotheses.extend(preds)

                assert len(references) == len(hypotheses)

                # Calculate BLEU-4 scores
                candidate = hypotheses[-1]
                reference = references[-1]
                references = []
                hypotheses = []

                refer = to_caps(reference, is_all_hot=False, word_map=word_map)
                caps = to_caps([candidate], is_all_hot=False, word_map=word_map)
                pri_image = pri_image.squeeze(0).tolist()
                pri_image = np.transpose(pri_image, (1, 2, 0))
                if save_img:
                    img_show(pri_image, caps, refer, is_path=False,
                             show=False, save_path=pre_save_path,
                             id=str(flag) + "_" + str(i))
                # Calculate  scores
                [b1, b2, b3, b4], b_avg = get_bleu([reference], [candidate], all_bleu=True)
                [r1, r2, r3, r4], r_avg = get_rouge([reference], [candidate], all_rouge=True)
                bleu.update(b_avg)
                bleu1.update(b1)
                bleu2.update(b2)
                bleu3.update(b3)
                bleu4.update(b4)
                rouge.update(r_avg)
                rouge1.update(r1['f'])
                rouge2.update(r2['f'])
                rouge3.update(r3['f'])
                rouge4.update(r4['f'])
                # Write validation metrics to TensorBoard
                if write_log:
                    writer.add_scalars('Test/bleu', {'bleu1': b1, 'bleu1_avg': bleu1.avg}, i)
                    writer.add_scalars('Test/bleu', {'bleu2': b2, 'bleu2_avg': bleu2.avg}, i)
                    writer.add_scalars('Test/bleu', {'bleu3': b3, 'bleu3_avg': bleu3.avg}, i)
                    writer.add_scalars('Test/bleu', {'bleu4': b4, 'bleu4_avg': bleu4.avg}, i)
                    writer.add_scalars('Test/bleu', {'bleu_avg(1-4)': bleu.avg}, i)

                    writer.add_scalars('Test/rouge',
                                       {'r1_F1': r1['f'], 'r1_precision': r1['p'],
                                        'r1_recall': r1['r'], 'r1_F1_avg': rouge1.avg}, i)
                    writer.add_scalars('Test/rouge',
                                       {'r2_F1': r2['f'], 'r2_precision': r2['p'],
                                        'r2_recall': r2['r'], 'r2_F1_avg': rouge2.avg}, i)
                    writer.add_scalars('Test/rouge',
                                       {'r3_F1': r3['f'], 'r3_precision': r3['p'],
                                        'r3_recall': r3['r'], 'r3_F1_avg': rouge3.avg}, i)
                    writer.add_scalars('Test/rouge',
                                       {'r4_F1': r4['f'], 'r4_precision': r4['p'],
                                        'r4_recall': r4['r'], 'r4_F1_avg': rouge4.avg}, i)
                    writer.add_scalars('Test/rouge', {'rouge_avg(1-4)': rouge.avg}, i)

                if eval_len and i >= eval_len:
                    print("Reached Max length")
                    return

                t.set_postfix(bleu=f'{bleu.val * 100.0:.3f}%({bleu.avg * 100.0:.3f})%',
                              rouge=f'{rouge.val * 100.0:.3f}%({rouge.avg * 100.0:.3f})%')
                t.update(1)


def evaluate(beam_size=5, max_len=50, write_log=False, eval_len=0, save_img=True, flag=""):
    global word_map
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    datasets_name = "flickr_use_premodel"
    data_folder = f'out_data/{datasets_name}/out_hdf5/per_5_freq_1_maxlen_50'  # folder with data files saved by create_input_files.py
    data_name = f'{datasets_name}_5_cap_per_img_1_min_word_freq'  # base name shared by data files

    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!
    references = list()
    hypotheses = list()

    bleu = AverageMeter()
    bleu1 = AverageMeter()
    bleu2 = AverageMeter()
    bleu3 = AverageMeter()
    bleu4 = AverageMeter()
    rouge = AverageMeter()
    rouge1 = AverageMeter()
    rouge2 = AverageMeter()
    rouge3 = AverageMeter()
    rouge4 = AverageMeter()

    # For each image
    print("Device: ", device)
    time.sleep(0.01)
    with tqdm(loader, desc='Evaluating at Beam Size ' + str(beam_size)) as t:
        for i, (image, caps, caplens, allcaps, pri_image) in enumerate(t):
            k = beam_size

            # Move to GPU device, if available
            image = image.to(device)  # (1, 3, 256, 256)

            # Encode
            encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

            enc_image_size = encoder_out.size(1)
            encoder_dim = encoder_out.size(3)

            # Flatten encoding
            encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1)

            # We'll treat the problem as having a batch size of k
            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

            # Tensor to store top k previous words at each step; now they're just <start>
            k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

            # Lists to store completed sequences and scores
            complete_seqs = list()
            complete_seqs_scores = list()

            # Start decoding
            step = 1
            h, c = decoder.init_hidden_state(encoder_out)

            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:

                embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
                gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe

                h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

                scores = decoder.fc(h)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                prev_word_inds = top_k_words // vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)
                # 添加边界检查
                prev_word_inds = torch.clamp(prev_word_inds, max=vocab_size - 1)
                next_word_inds = torch.clamp(next_word_inds, max=vocab_size - 1)

                # TODO
                # Add new words to sequences
                # seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1).long()], dim=1)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                elif step > max_len:
                    complete_seqs.extend(seqs.tolist())
                    complete_seqs_scores.extend(top_k_scores)
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                # Break if things have been going on too long
                if step > max_len:
                    break
                step += 1

            k = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[k]

            # References
            img_caps = allcaps[0].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)

            # Hypotheses
            hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

            assert len(references) == len(hypotheses)

            candidate = hypotheses[-1]
            reference = references[-1]

            refer = to_caps(reference, is_all_hot=False, word_map=word_map)
            caps = to_caps([candidate], is_all_hot=False, word_map=word_map)
            pri_image = pri_image.squeeze(0).tolist()
            pri_image = np.transpose(pri_image, (1, 2, 0))
            if save_img:
                img_show(pri_image, caps, refer, is_path=False,
                         show=False, save_path=pre_save_path,
                         id=str(flag) + "_" + str(i))
            # Calculate  scores
            [b1, b2, b3, b4], b_avg = get_bleu([reference], [candidate], all_bleu=True)
            [r1, r2, r3, r4], r_avg = get_rouge([reference], [candidate], all_rouge=True)
            bleu.update(b_avg)
            bleu1.update(b1)
            bleu2.update(b2)
            bleu3.update(b3)
            bleu4.update(b4)
            rouge.update(r_avg)
            rouge1.update(r1['f'])
            rouge2.update(r2['f'])
            rouge3.update(r3['f'])
            rouge4.update(r4['f'])
            # Write validation metrics to TensorBoard
            if write_log:
                writer.add_scalars('Test/bleu', {'bleu1': b1, 'bleu1_avg': bleu1.avg}, i)
                writer.add_scalars('Test/bleu', {'bleu2': b2, 'bleu2_avg': bleu2.avg}, i)
                writer.add_scalars('Test/bleu', {'bleu3': b3, 'bleu3_avg': bleu3.avg}, i)
                writer.add_scalars('Test/bleu', {'bleu4': b4, 'bleu4_avg': bleu4.avg}, i)
                writer.add_scalars('Test/bleu', {'bleu_avg(1-4)': bleu.avg}, i)

                writer.add_scalars('Test/rouge',
                                   {'r1_F1': r1['f'], 'r1_precision': r1['p'],
                                    'r1_recall': r1['r'], 'r1_F1_avg': rouge1.avg}, i)
                writer.add_scalars('Test/rouge',
                                   {'r2_F1': r2['f'], 'r2_precision': r2['p'],
                                    'r2_recall': r2['r'], 'r2_F1_avg': rouge2.avg}, i)
                writer.add_scalars('Test/rouge',
                                   {'r3_F1': r3['f'], 'r3_precision': r3['p'],
                                    'r3_recall': r3['r'], 'r3_F1_avg': rouge3.avg}, i)
                writer.add_scalars('Test/rouge',
                                   {'r4_F1': r4['f'], 'r4_precision': r4['p'],
                                    'r4_recall': r4['r'], 'r4_F1_avg': rouge4.avg}, i)
                writer.add_scalars('Test/rouge', {'rouge_avg(1-4)': rouge.avg}, i)

            if eval_len and i >= eval_len:
                return

            t.set_postfix(bleu=f'{bleu.val * 100.0:.3f}%({bleu.avg * 100.0:.3f})%',
                          rouge=f'{rouge.val * 100.0:.3f}%({rouge.avg * 100.0:.3f})%')


if __name__ == '__main__':
    # DataLoader
    datasets_name = "coco_use_premodel"
    data_folder = f'out_data/{datasets_name}/out_hdf5/per_5_freq_1_maxlen_50'  # folder with data files saved by create_input_files.py
    data_name = f'{datasets_name}_5_cap_per_img_1_min_word_freq'  # base name shared by data files

    pre_save_path = f'pre_img/{datasets_name}/'
    pre_save_path, _, _ = path_checker(pre_save_path, False, True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().strftime("%Y%m%d")
    log_dir = fr'.\logs\{datasets_name}\Test\{datasets_name}_{timestamp}'
    log_write(log_dir, log_dir)
    writer = SummaryWriter(log_dir)

    # Load word map (word2ix)
    word_map_path = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    word_map_path = os.path.normpath(word_map_path)

    with open(word_map_path, 'r') as j:
        word_map = json.load(j)

    # rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    model_list = [
        "out_data/coco_use_premodel/save_model/thesis/checkpoint_coco_use_premodel_5_cap_per_img_1_min_word_freq_epoch_0.pth",
        "out_data/coco_use_premodel/save_model/thesis/temp_checkpoint_coco_use_premodel_5_cap_per_img_1_min_word_freq_epoch_0_batch_5900.pth",
        "out_data/coco_use_premodel/save_model/thesis/temp_checkpoint_coco_use_premodel_5_cap_per_img_1_min_word_freq_epoch_0_batch_11796.pth",
        "out_data/coco_use_premodel/save_model/other/epoch_0_lr_4e-4_batch_32_att_1024_de_512_drop_0.3_gama_0.95/temp_checkpoint_coco_use_premodel_5_cap_per_img_1_min_word_freq_epoch_0_batch_5940.pth",
        "out_data/coco_use_premodel/save_model/other/epoch_0_lr_4e-4_batch_32_att_1024_de_512_drop_0.3_gama_0.95/temp_checkpoint_coco_use_premodel_5_cap_per_img_1_min_word_freq_epoch_0_batch_11957.pth",
        "out_data/coco_use_premodel/save_model/other/lr_4e-4_batch_32_att_512_de_512_drop_0.5_gama_0.95/BEST_checkpoint_coco_use_premodel_5_cap_per_img_1_min_word_freq_epoch_0.pth",
        "out_data/coco_use_premodel/save_model/other/lr_4e-4_batch_32_att_512_de_512_drop_0.5_gama_0.95/BEST_checkpoint_coco_use_premodel_5_cap_per_img_1_min_word_freq_epoch_1.pth",
        "out_data/coco_use_premodel/save_model/other/lr_4e-4_batch_32_att_512_de_512_drop_0.5_gama_0.95/temp_checkpoint_coco_use_premodel_5_cap_per_img_1_min_word_freq_epoch_1_batch_5888.pth",
        "out_data/coco_use_premodel/save_model/other/lr_4e-4_batch_32_att_512_de_512_drop_0.5_gama_0.95/temp_checkpoint_coco_use_premodel_5_cap_per_img_1_min_word_freq_epoch_1_batch_8292.pth"
    ]

    test_name = "flickr_use_premodel"
    test_data_folder = f'out_data/{test_name}/out_hdf5/per_5_freq_1_maxlen_50'  # folder with data files saved by create_input_files.py
    test_data_name = f'{test_name}_5_cap_per_img_1_min_word_freq'  #

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(test_data_folder, test_data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    for i, checkpoint in enumerate(model_list):
        # Load model
        checkpoint = torch.load(checkpoint)
        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        encoder = checkpoint['encoder']
        encoder = encoder.to(device)
        encoder.eval()
        evaluate(beam_size=5, eval_len=15, save_img=True, flag=str(i))

        # eval_validate(val_loader, encoder, decoder, word_map, write_log=False, eval_len=10, save_img=True,
        #               flag="coco_" + str(i))
