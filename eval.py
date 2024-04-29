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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lots of computational overhead

datasets_name = 'flickr'

data_folder = f'out_data/{datasets_name}/out_hdf5/per_5_freq_5_maxlen_18'  # folder with data files saved by create_input_files.py
data_name = f'{datasets_name}_5_cap_per_img_5_min_word_freq'  # base name shared by data files
model_save_path = f'out_data/{datasets_name}/save_model'


pre_save_path = f'out_data/{datasets_name}/pre_img/'
pre_save_path,_,_ = path_checker(pre_save_path,False,True)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamp = datetime.now().strftime("%Y%m%d")
log_dir = fr'.\logs\{datasets_name}\Test\epoch_0_{timestamp}'
writer = SummaryWriter(log_dir)

model_name = 'checkpoint_flickr_5_cap_per_img_5_min_word_freq_epoch_0.pth'
checkpoint = f'{model_save_path}/{model_name}'  # model checkpoint

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
word_map_file = os.path.normpath(word_map_file)
try:
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
except Exception as e:
    print(Fore.YELLOW + "\nError eval(), read word_map =>", e)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size=5, max_len=35):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!
    references = list()
    hypotheses = list()

    bleu = AverageMeter()
    rouge = AverageMeter()

    # For each image

    with tqdm(loader, desc='EVALUATING AT BEAM SIZE ' + str(beam_size)) as t:
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

            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

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
            img_show(pri_image, caps, refer, False,pre_save_path,i)

            # Calculate  scores
            bleu.update(get_bleu([reference], [candidate]))
            rouge.update(get_rouge([reference], [candidate]))
            writer.add_scalars('Test/bleu', {'val': bleu.val, 'avg': bleu.avg}, i)
            writer.add_scalars('Test/rouge', {'val': rouge.val, 'avg': rouge.avg}, i)

            t.set_postfix(bleu=f'{bleu.val * 100.0:.3f}({bleu.avg * 100.0:.3f})',
                          rouge=f'{rouge.val * 100.0:.3f}({rouge.avg * 100.0:.3f})')


if __name__ == '__main__':
    pass
    # beam_size = 1
    evaluate()
    # bleu,rouge = evaluate(beam_size)
    # print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size,bleu))
    # print("\nRouge score @ beam size of %d is %.4f." % (beam_size,rouge))
