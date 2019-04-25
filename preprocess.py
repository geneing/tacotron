import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import blizzard, ljspeech, mailabs
from hparams import hparams


def preprocess_blizzard(args):
  in_dir = os.path.join(args.base_dir, 'Blizzard2012')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = blizzard.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def preprocess_ljspeech(args):
  in_dir = os.path.join(args.base_dir, 'LJSpeech-1.1')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = ljspeech.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)

def preprocess_mailabs(args):
  in_dir = os.path.expanduser(args.base_dir)
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = mailabs.build_from_path(in_dir, out_dir, args, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
  with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    for m in metadata:
      f.write('|'.join([str(x) for x in m]) + '\n')
  frames = sum([m[2] for m in metadata])
  hours = frames * hparams.frame_shift_ms / (3600 * 1000)
  print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
  print('Max input length:  %d' % max(len(m[3]) for m in metadata))
  print('Max output length: %d' % max(m[2] for m in metadata))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('.'))
  parser.add_argument('--output', default='training')
  parser.add_argument('--dataset', required=True, choices=['blizzard', 'ljspeech','M-AILABS'])
  parser.add_argument('--num_workers', type=int, default=cpu_count())
  parser.add_argument('--language', default='en_US')
  parser.add_argument('--voice', default='female')
  parser.add_argument('--reader', default='mary_ann')
  parser.add_argument('--merge_books', default='False')
  parser.add_argument('--book', default='northandsouth')
  args = parser.parse_args()
  if args.dataset == 'blizzard':
    preprocess_blizzard(args)
  elif args.dataset == 'ljspeech':
    preprocess_ljspeech(args)
  elif args.dataset == 'M-AILABS':
    preprocess_mailabs(args)


if __name__ == "__main__":
  main()
