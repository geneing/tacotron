from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from util import audio

def norm_data(args):
  
  merge_books = (args.merge_books=='True')

  supported_languages = ['en_US', 'en_UK', 'fr_FR', 'it_IT', 'de_DE', 'es_ES', 'ru_RU',
                         'uk_UK', 'pl_PL', 'nl_NL', 'pt_PT', 'fi_FI', 'se_SE', 'tr_TR', 'ar_SA']
  if args.language not in supported_languages:
    raise ValueError('Please enter a supported language to use from M-AILABS dataset! \n{}'.format(
      supported_languages))
  
  supported_voices = ['female', 'male', 'mix']
  if args.voice not in supported_voices:
    raise ValueError('Please enter a supported voice option to use from M-AILABS dataset! \n{}'.format(
      supported_voices))
  
  path = os.path.join(args.base_dir, args.language, 'by_book', args.voice)
  supported_readers = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path,e))]
  if args.reader not in supported_readers:
    raise ValueError('Please enter a valid reader for your language and voice settings! \n{}'.format(
      supported_readers))
  
  path = os.path.join(path, args.reader)
  supported_books = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path,e))]
  if merge_books:
    return [os.path.join(path, book) for book in supported_books]
  
  else:
    if args.book not in supported_books:
      raise ValueError('Please enter a valid book for your reader settings! \n{}'.format(
        supported_books))
    
    return [os.path.join(path, args.book)]


def build_from_path(in_dir, out_dir, args, tqdm=lambda x: x):
  '''Preprocesses the M-AILABS dataset from a given input path into a given output directory.

    Args:
      in_dir: The root directory for the M-AILABS dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallelize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  num_workers = args.num_workers
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1
  
  input_folders = norm_data(args)
  for input_dir in input_folders:
    with open(os.path.join(input_dir, 'metadata.csv'), encoding='utf-8') as f:
      for line in f:
        parts = line.strip().split('|')
        wav_path = os.path.join(input_dir, 'wavs', '%s.wav' % parts[0])
        text = parts[2]
        futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
        index += 1
  return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(out_dir, index, wav_path, text):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    text: The text spoken in the input audio file

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''

  # Load the audio to a numpy array:
  
  try:
    # Load the audio as numpy array
    wav = audio.load_wav(wav_path)
  except FileNotFoundError: #catch missing wav exception
    print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
      wav_path))
    return None
  
  # Compute the linear-scale spectrogram from the wav:
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]

  # Compute a mel-scale spectrogram from the wav:
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  # Write the spectrograms to disk:
  spectrogram_filename = 'ljspeech-spec-%05d.npy' % index
  mel_filename = 'ljspeech-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example:
  return (spectrogram_filename, mel_filename, n_frames, text)
